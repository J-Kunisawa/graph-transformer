import math
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing

# DGL Sparseライブラリのインポートを試みる（ sparseなアテンションや行列演算の高速化用 ）
try:
    import dgl.sparse as dglsp
    from dgl.sparse import SparseMatrix

    HAS_DGL_SPARSE = True
except ImportError:
    HAS_DGL_SPARSE = False
    dglsp = None
    SparseMatrix = Any  # DGLが無い場合はAnyをプレースホルダーとして使用


# --- グローバルな訓練最適化フラグの管理 ---
_MEMORY_OPTIMIZED_TRAINING: bool = False


def set_memory_optimized_training(enabled: bool) -> None:
    """
    メモリ最適化トレーニング機能（AMPセーフな疎行列演算 + アクティベーション・チェックポイント）の有効/無効を切り替える。
    """
    global _MEMORY_OPTIMIZED_TRAINING
    _MEMORY_OPTIMIZED_TRAINING = bool(enabled)


def use_memory_optimized_training() -> bool:
    """
    グローバルなメモリ最適化トレーニングが有効な場合は True を返す。
    """
    return _MEMORY_OPTIMIZED_TRAINING


@contextmanager
def _no_cuda_autocast_if_needed(is_cuda_tensor: bool):
    """
    特定の計算カーネルで数値的不安定（アンダーフロー等）を防ぐため、
    一時的にCUDAの自動混合精度（Autocast / 16bit）を無効化し、FP32（単精度）を強制するコンテキストマネージャ。
    """
    if is_cuda_tensor and torch.is_autocast_enabled():
        with torch.amp.autocast("cuda", enabled=False):
            yield
    else:
        yield


def _bsddmm_fp32(mask, q: torch.Tensor, k_t: torch.Tensor):
    """
    DGLの「サンプル付き高密度行列積 (Sampled Dense-Dense Matrix Multiplication)」を
    強制的にFP32で安全に実行し、稀なアテンションテンソル（疎行列）を返す。
    """
    q32 = q.float()
    k_t32 = k_t.float()
    try:
        mask32 = mask.astype(torch.float32)
    except AttributeError:
        if hasattr(mask, "val") and mask.val.dtype != torch.float32:
            if hasattr(dglsp, "val_like"):
                mask32 = dglsp.val_like(mask, mask.val.float())
            else:
                mask32 = dglsp.spmatrix(mask.indices(), mask.val.float(), mask.shape)
        else:
            mask32 = mask
    with _no_cuda_autocast_if_needed(q.is_cuda):
        return dglsp.bsddmm(mask32, q32, k_t32)


def _bspmm_fp32(attn, v: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    DGLの「疎行列と高密度行列の積 (Sparse-Dense Matrix Multiplication)」を
    FP32で実行し、得られた高密度テンソルを出力用の元のデータ型（out_dtype）にキャストして返す。
    """
    v32 = v.float()
    try:
        attn32 = attn.astype(torch.float32)
    except AttributeError:
        if hasattr(attn, "val") and attn.val.dtype != torch.float32:
            if hasattr(dglsp, "val_like"):
                attn32 = dglsp.val_like(attn, attn.val.float())
            else:
                attn32 = dglsp.spmatrix(attn.indices(), attn.val.float(), attn.shape)
        else:
            attn32 = attn
    with _no_cuda_autocast_if_needed(v.is_cuda):
        y32 = dglsp.bspmm(attn32, v32)
    return y32.to(out_dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    LayerNormの平均計算を省略し、2乗平均平方根のみを用いて特徴量の最終次元に沿って標準化を行う高速な正規化レイヤー。
    """
    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        """
        Args:
            d (int): 入力テンソルの特徴次元数。
            p (float, optional): Partial RMSNorm の割合。 [0, 1] の値を指定すると、全チャンネルの
                                 うち上位数％のみの統計量で代表させてスケーリングを計算する。-1.0で無効。
            eps (float, optional): ゼロ除算を避けるための微小値。
            bias (bool, optional): バイアス項（offset）を含めるかどうか。
        """
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d)) # 学習可能な拡大パラメータ (Gamma)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d)) # 学習可能な移動パラメータ (Beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # p が 0〜1 の範囲外の場合は、通常通り全次元(d)でRMSを計算
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            # 特徴量の一部のチャンネル(次元)のみを切り出してRMS統計量を算出（Partial RMSNormによる高速化）
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        # 2乗平均平方根を算出して正規化
        rms_x = norm_x / math.sqrt(d_x)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class IdentityRMSNorm(RMSNorm):
    """state_dict互換性を保ったまま正規化をバイパスするRMSNorm。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scaleをゼロ係数で接続し、再開時のoptimizer parameter順序も維持する。
        return x + self.scale.sum().to(dtype=x.dtype) * 0.0


# --- MLP用の活性化関数の設定 ---
_USE_SILU_ACTIVATION: bool = False


def set_use_silu_activation(use_silu: bool) -> None:
    """MLP構築時のデフォルト活性化関数として SiLU (Swish) を強制するかを設定。"""
    global _USE_SILU_ACTIVATION
    _USE_SILU_ACTIVATION = use_silu


def use_silu_activation() -> bool:
    """SiLU活性化関数がグローバルに有効化されている場合は True を返す。"""
    return _USE_SILU_ACTIVATION


def _resolve_activation(act: Optional[str]) -> str:
    if act is None:
        return "silu" if _USE_SILU_ACTIVATION else "relu"
    return act


ACTIVATION = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: Optional[str] = None,
) -> nn.Module:
    """
    多層パーセプトロン (MLP) を構築するヘルパー関数。
    MeshGraphNetのEncoder/Decoder/Processorブロックの各所で特徴量変換に多用される。
    """
    assert nb_of_layers >= 2, "MLPは少なくとも2レイヤー（入力層と出力層）以上で構成される必要があります。"

    act_key = _resolve_activation(act)
    if act_key not in ACTIVATION:
        raise NotImplementedError(f"活性化関数 '{act_key}' はサポートされていません。利用可能: {list(ACTIVATION)}")
    activation = ACTIVATION[act_key]

    # 入力層の定義
    layers = [nn.Linear(in_size, hidden_size), activation()]

    # 隠れ層（中間層）の追加
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size), activation()])

    # 出力層の追加
    layers.append(nn.Linear(hidden_size, out_size))

    # オプションで最終層に RMSNorm を追加して出力を安定させる
    if layer_norm:
        layers.append(RMSNorm(out_size))

    return nn.Sequential(*layers)


class GatedMLP(nn.Module):
    """
    Gated MLP レイヤー。
    入力を2つの線形層に分岐させ、片方に活性化関数を適用したあと、もう片方と要素積（Gating）をとる構造。
    近年、TransformerのFeed-Forwardネットワークにおいて精度向上のためによく使われる。
    """
    def __init__(self, in_size: int, hidden_size: int, expansion_factor: int):
        super().__init__()
        # 中間層のチャンネルサイズを拡大（expansion_factor倍）して並列にマッピング
        self.linear1 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.linear2 = nn.Linear(in_size, expansion_factor * hidden_size)

        activation_cls = nn.SiLU if use_silu_activation() else nn.GELU
        self.activation = activation_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 片方のブランチ（left）にのみゲートをかける活性化関数を適用し、乗算して出力を制御する
        left = self.activation(self.linear1(x))
        right = self.linear2(x)
        return left * right


def build_gated_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    expansion_factor: int = 3,
    layer_norm: bool = True,
) -> nn.Module:
    """Gated MLPブロックを構築し、必要な場合だけ入力RMSNormを追加する。"""
    layers = [
        RMSNorm(in_size) if layer_norm else IdentityRMSNorm(in_size),
        GatedMLP(in_size=in_size, hidden_size=hidden_size, expansion_factor=expansion_factor),
        nn.Linear(hidden_size * expansion_factor, out_size),
    ]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """
    オンライン統計情報更新型ノーマライザー。
    訓練データ（流速場、圧力場など）のスケールを、学習中にリアルタイムに移動平均・移動分散として蓄積し、
    平均0、標準偏差1の範囲にデータを標準化（正規化）する。物理AIの学習の安定化に必須のモジュール。
    """
    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: Optional[Union[str, torch.device]] = "cuda",
    ):
        super().__init__()
        self.name = name
        self.device = device
        self._max_accumulations = max_accumulations # 数値的な不安定化を防ぐための、最大統計蓄積ステップ数
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        
        # PyTorchのregister_bufferを使い、モデルの重み(Parameter)ではないが、
        # 状態として保存・ロード（checkpointのstate_dictに含まれる）すべき統計バッファを登録
        self.register_buffer("_acc_count", torch.tensor(0.0, device=device)) # 処理した総ノード（データサンプル）数
        self.register_buffer("_num_accumulations", torch.tensor(0.0, device=device)) # 蓄積回数のカウント
        self.register_buffer("_acc_sum", torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))
        self.register_buffer("_acc_sum_squared", torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))

    def forward(self, batched_data: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        """データを正規化し、必要に応じて現在のバッチデータから統計情報を更新（蓄積）する。"""
        if accumulate:
            # 指定した最大ステップ数に達するまで統計を更新し続ける
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        # (X - Mean) / Std の標準化を実行
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """正規化されたデータを、元の物理的なスケール（m/sやPaなど）に逆変換（復元）する。"""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        """現在のミニバッチの和および2乗和を計算し、グローバルバッファに加算する（オンライン更新アルゴリズム）。"""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data**2, dim=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self) -> torch.Tensor:
        """蓄積された合計値から、現在の平均値を算出する。"""
        safe_count = torch.max(self._acc_count, torch.tensor(1.0, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        """蓄積された2乗和と平均値から分散を求め、標準偏差（Std）を算出する。"""
        safe_count = torch.max(self._acc_count, torch.tensor(1.0, device=self._acc_count.device))
        variance = self._acc_sum_squared / safe_count - self._mean() ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.max(std, self._std_epsilon) # ゼロ除算を防ぐためepsilonでクリップ

    def get_variable(self) -> Dict[str, Any]:
        """ノーマライザーの現在の統計情報をデバッグ等のために辞書として取得する。"""
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }


# --- RoPE (Rotary Positional Embeddings) の空間拡張用のユーティリティ ---

def _make_inv_freq(m: int, base: float, device: torch.device) -> torch.Tensor:
    """RoPEで周波数軸をマッピングするための逆周波数ベクトル（inverse frequencies）を事前計算する。"""
    if m <= 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    step = math.log(base) / max(m, 1)
    return torch.exp(-torch.arange(m, device=device, dtype=torch.float32) * step)


def _apply_rope_with_inv(
    q: torch.Tensor,
    k: torch.Tensor,
    pos: torch.Tensor,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TransformerのQ（Query）およびK（Key）テンソルに対して、3次元等の空間座標（pos）に基づく
    回転式位置埋め込み（RoPE）を適用する。メッシュ上の幾何学的な位置関係をアテンション計算に組み込む。
    """
    N, D, H = q.shape
    pos_dimension = pos.shape[1] # 空間の次元（2Dなら2, 3Dなら3）
    m = D // (pos_dimension * 2)
    if m == 0 or inv_freq.numel() == 0:
        return q, k

    d_rope = pos_dimension * 2 * m
    q_dtype = q.dtype

    # 3D位置座標と周波数を外積の形で掛け合わせ、各軸ごとの回転角（角度場）を計算
    pos_f32 = pos[:, :pos_dimension].to(torch.float32)
    inv_freq_f32 = inv_freq.to(pos.device, dtype=torch.float32)
    angles = pos_f32.unsqueeze(-1) * inv_freq_f32.view(1, 1, m)

    if hasattr(torch, "sincos"):
        sin_f32, cos_f32 = torch.sincos(angles)
    else:
        cos_f32, sin_f32 = torch.cos(angles), torch.sin(angles)

    sin = sin_f32.to(dtype=q_dtype, device=q.device)
    cos = cos_f32.to(dtype=q_dtype, device=q.device)

    # テンソルを偶数・奇数インデックスに分解して2次元平面上での複素数回転（RoPEアルゴリズム）を適用
    def _apply(x: torch.Tensor) -> torch.Tensor:
        part = (
            x[:, :d_rope, :]
            .contiguous()
            .view(N, pos_dimension, 2 * m, H)
            .view(N, pos_dimension, m, 2, H)
        )
        rest = x[:, d_rope:, :]

        even = part[..., 0, :]
        odd = part[..., 1, :]

        cos_b = cos.unsqueeze(-1)
        sin_b = sin.unsqueeze(-1)

        # 2次元の回転行列 [ [cos, -sin], [sin, cos] ] を掛け合わせる操作に相当
        rot_even = even * cos_b - odd * sin_b
        rot_odd = even * sin_b + odd * cos_b

        rot = (
            torch.stack((rot_even, rot_odd), dim=3)
            .reshape(N, pos_dimension, 2 * m, H)
            .reshape(N, d_rope, H)
        )

        out = torch.empty_like(x)
        out[:, :d_rope, :] = rot
        if D > d_rope:
            out[:, d_rope:, :] = rest
        return out

    return _apply(q), _apply(k)


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask,
) -> torch.Tensor:
    """Scaled Dot-Product AttentionのQuery-Key乗算およびSoftmaxステップ。マスク付きの疎行列計算に対応。"""
    scaling_factor = math.sqrt(k.size(1))
    q = q / scaling_factor # スケーリング因数（√d_k）での除算

    # DGL Sparseが有効で、隣接行列（att_mask）が与えられている場合は疎行列アテンションを実行
    if att_mask is not None and HAS_DGL_SPARSE:
        if hasattr(att_mask, "val") and att_mask.val.dtype != q.dtype:
            if hasattr(dglsp, "val_like"):
                att_mask = dglsp.val_like(att_mask, att_mask.val.to(q.dtype))
            else:
                # Fallback to reconstructing if val_like is missing
                att_mask = dglsp.spmatrix(att_mask.indices(), att_mask.val.to(q.dtype), att_mask.shape)

        if _MEMORY_OPTIMIZED_TRAINING and q.is_cuda and torch.is_autocast_enabled():
            # メモリ最適化モードならAMPセーフな強制FP32カーネルを使用
            attn = _bsddmm_fp32(att_mask, q, k.transpose(1, 0))
        else:
            attn = dglsp.bsddmm(att_mask, q, k.transpose(1, 0))
        attn = attn.softmax() # 接続されているエッジのみに沿ってSoftmaxを計算
    else:
        # メッシュ外も含めて総当たりでアテンションを張る標準的な密行列計算
        if q.ndim == 3:
            # q/k: [node, head_dim, head] -> [query_node, key_node, head]
            attn = torch.einsum("idh,jdh->ijh", q, k)
            attn = torch.softmax(attn, dim=1)
        else:
            attn = q @ k.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)

    return attn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask=None,
    return_attention: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Scaled Dot-Product Attentionの統合実行関数（Q, K, Vから文脈ベクトルへの集約までを行う）。"""
    if isinstance(att_mask, (list, tuple)):
        if q.ndim < 3:
            raise ValueError("head別マスクにはhead次元を持つQ/K/Vが必要です。")
        num_heads = q.size(-1)
        if len(att_mask) != num_heads:
            raise ValueError(
                f"head別マスク数({len(att_mask)})はhead数({num_heads})と一致する必要があります。"
            )

        # 同一マスクを使うheadをまとめ、DGL sparse kernelの呼び出し回数を抑える。
        grouped_masks = {}
        for head_index, mask in enumerate(att_mask):
            key = id(mask)
            if key not in grouped_masks:
                grouped_masks[key] = (mask, [])
            grouped_masks[key][1].append(head_index)

        outputs = [None] * num_heads
        attention_groups = []
        for mask, head_indices in grouped_masks.values():
            index = torch.as_tensor(head_indices, dtype=torch.long, device=q.device)
            grouped_output, grouped_attention = scaled_dot_product_attention(
                q.index_select(-1, index),
                k.index_select(-1, index),
                v.index_select(-1, index),
                att_mask=mask,
                return_attention=True,
            )
            for local_index, head_index in enumerate(head_indices):
                outputs[head_index] = grouped_output[..., local_index : local_index + 1]
            attention_groups.append((tuple(head_indices), grouped_attention))

        output = torch.cat(outputs, dim=-1)
        if return_attention:
            return output, attention_groups
        return output

    attn = scaled_query_key_softmax(q, k, att_mask=att_mask)

    # アテンション重みとValue（V）の行列積
    if att_mask is not None and HAS_DGL_SPARSE:
        if _MEMORY_OPTIMIZED_TRAINING and v.is_cuda and torch.is_autocast_enabled():
            y = _bspmm_fp32(attn, v, v.dtype)
        else:
            y = dglsp.bspmm(attn, v)
    else:
        if attn.ndim == 3 and v.ndim == 3:
            y = torch.einsum("ijh,jdh->idh", attn, v)
        else:
            y = attn @ v

    if return_attention:
        return y, attn
    else:
        return y


class Attention(nn.Module):
    """マルチヘッド・自己アテンション（Multi-Head Attention）レイヤー。空間RoPEおよびGating機構を内包。"""
    def __init__(
        self,
        input_dim=512,
        output_dim=512,
        num_heads=4,
        pos_dimension: int = 3,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        assert output_dim % num_heads == 0, "出力次元数はアテンションヘッド数で割り切れる必要があります。"

        self.hidden_size = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.pos_dimension = pos_dimension
        self.rope_base = rope_base

        # Q, K, V への各射影線形層
        self.q_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.v_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.proj = nn.Linear(output_dim, output_dim, bias=use_proj_bias) # 最終統合層

        # 空間RoPEが有効な場合は周波数を事前計算してバッファ登録
        if self.use_rope_embeddings:
            self.m = self.head_dim // max(self.pos_dimension * 2, 1)
            inv = _make_inv_freq(self.m, self.rope_base, torch.device("cpu"))
            self.register_buffer("rope_inv_freq", inv, persistent=True)
        else:
            self.m = 0
            self.register_buffer("rope_inv_freq", torch.empty(0, dtype=torch.float32), persistent=False)

        # ゲート付きアテンション用の射影層
        if self.use_gated_attention:
            self.gate_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        else:
            self.gate_proj = None

        # 計算高速化のため、Q, K, V の初期重みを共有（パラメータ共通化）する設計オプション
        if not use_separate_proj_weight:
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

    def forward(
        self,
        x: torch.Tensor,
        adj,
        pos: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if self.use_rope_embeddings and pos is None:
            raise ValueError("RoPE位置埋め込みが有効な場合、ノードの位置座標情報（pos）が必須です。")

        N = x.size(0)
        query, key, value = x, x, x

        # Q, K, V への線形写像
        q, k, v = map(lambda fn, t: fn(t), [self.q_proj, self.k_proj, self.v_proj], [query, key, value])

        # マルチヘッド用にテンソルの形状を変更 [NumNodes, HeadDim, NumHeads]
        q = q.reshape(N, self.head_dim, self.num_heads)
        k = k.reshape(N, self.head_dim, self.num_heads)
        v = v.reshape(N, self.head_dim, self.num_heads)

        # 空間RoPE（回転）の適用
        if self.use_rope_embeddings and self.rope_inv_freq.numel() > 0:
            q, k = _apply_rope_with_inv(q, k, pos, self.rope_inv_freq)

        # アテンション計算の実行
        if return_attention:
            y, attn = scaled_dot_product_attention(q, k, v, adj, return_attention=True)
        else:
            y = scaled_dot_product_attention(q, k, v, adj)

        # アテンションの出力に対する Sigmoid ゲート制御（Gating）
        if self.use_gated_attention and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x)).reshape(N, self.head_dim, self.num_heads)
            gate = gate.to(dtype=y.dtype, device=y.device)
            y = y * gate

        # ヘッドを結合して全結合層に通す
        out = self.proj(y.reshape(N, -1))

        if return_attention:
            return out, attn
        else:
            return out


class Transformer(nn.Module):
    """
    1つのトランスフォーマーブロック。
    [ 正規化 ➔ Sparse自己アテンション ➔ 残差結合 ] ➔ [ 正規化 ➔ GatedMLP ➔ 残差結合 ]
    という標準的かつ強力なシミュレーション特徴量更新レイヤーを構成。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation_layer: torch.nn.Module = nn.ReLU,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        pos_dimension: int = 3,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.pos_dimension = pos_dimension

        # アテンションサブブロックの定義
        self.attention = Attention(
            input_dim=input_dim, output_dim=output_dim, num_heads=num_heads,
            pos_dimension=pos_dimension, use_proj_bias=use_proj_bias,
            use_separate_proj_weight=use_separate_proj_weight,
            use_rope_embeddings=use_rope_embeddings, use_gated_attention=use_gated_attention,
            rope_base=rope_base,
        )

        self.activation = activation_layer()
        self.norm1, self.norm2 = RMSNorm(output_dim), RMSNorm(output_dim) # 各残差前の前置正規化（Pre-LN）
        # norm2 がPre-Normを担当するため、Gated MLP内部のRMSNormは無効にする。
        self.gated_mlp = build_gated_mlp(
            in_size=output_dim,
            hidden_size=output_dim,
            out_size=output_dim,
            layer_norm=False,
        )

        self.use_adjacency = HAS_DGL_SPARSE
        self.use_activation_checkpointing = _MEMORY_OPTIMIZED_TRAINING # 省メモリ訓練フラグの同期

    def forward(
        self,
        x: torch.Tensor,
        adj,
        pos: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if not self.use_adjacency:
            adj = None

        if self.use_rope_embeddings and pos is None:
            raise ValueError("use_rope_embeddings=True の場合、ノード座標(pos)が必要です。")

        # アテンション重みそのものを可視化等で外部に返す場合（チェックポイントは不適用）
        if return_attention:
            x_, attn = self.attention(self.norm1(x), adj, pos=pos, return_attention=True)
            x = x + x_
            x = x + self.gated_mlp(self.norm2(x))
            return x, attn

        # 訓練中かつメモリ最適化が有効な場合、アクティベーション・チェックポイントを適用（VRAMを大幅に節約）
        if self.training and self.use_activation_checkpointing and torch.is_grad_enabled():
            def _attn_only(tensor_x: torch.Tensor) -> torch.Tensor:
                return self.attention(self.norm1(tensor_x), adj, pos=pos, return_attention=False)

            def _mlp_only(tensor_x: torch.Tensor) -> torch.Tensor:
                return self.gated_mlp(self.norm2(tensor_x))

            # 計算グラフの一部を捨て、逆伝播時に再計算（use_reentrant=False で最新のPyTorch仕様に対応）
            x = x + checkpoint(_attn_only, x, use_reentrant=False)
            x = x + checkpoint(_mlp_only, x, use_reentrant=False)
        else:
            # 通常の順伝播計算（推論時や、省メモリモード無効時）
            x = x + self.attention(self.norm1(x), adj, pos=pos, return_attention=False)
            x = x + self.gated_mlp(self.norm2(x))

        return x


class TemporalAttention(nn.Module):
    """
    時間的補正アテンション（Temporal Corrector）。
    現ステップの予測状態を Query/Value とし、過去ステップ(t-1)の物理特徴量を Key として
    クロスアテンション（交差アテンション）を解くことで、時間発展の物理的一貫性を補正するレイヤー。
    """
    def __init__(self, hidden_size: int, num_heads: int = 4, use_gate: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size はヘッド数で割り切れる必要があります。"
        self.h = hidden_size
        self.H = num_heads
        self.d = hidden_size // num_heads
        self.use_gate = use_gate

        self.q_proj = nn.Linear(self.h, self.h, bias=True)
        self.k_proj = nn.Linear(self.h, self.h, bias=True)
        self.v_proj = nn.Linear(self.h, self.h, bias=True)
        self.out_proj = nn.Linear(self.h, self.h, bias=True)

        # 時間変化の重要度を制御するタイムゲート
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * self.h, self.h), nn.SiLU(),
                nn.Linear(self.h, self.h), nn.Sigmoid(),
            )

        # 過去と現在の特徴量を融合するミキサー層
        self.mixer = nn.Sequential(
            nn.Linear(2 * self.h, self.h), nn.SiLU(),
            nn.Linear(self.h, self.h),
        )

    def forward(
        self,
        h_prev: torch.Tensor,  # 過去のステップ(t-1)の特徴量: [N, H]
        h_pred: torch.Tensor,  # モデルが予測した現在の特徴量: [N, H]
        adj: Optional[Any] = None,
    ) -> torch.Tensor:
        N = h_prev.size(0)

        # 各ブランチへの射影とマルチヘッド展開
        q = self.q_proj(h_pred).reshape(N, self.d, self.H)
        k = self.k_proj(h_prev).reshape(N, self.d, self.H)
        v = self.v_proj(h_pred).reshape(N, self.d, self.H)

        # 過去の情報（Key）を現在（Query）から参照して集約
        y = scaled_dot_product_attention(q, k, v, adj)
        out = self.out_proj(y.reshape(N, self.h))

        # 過去と現在の履歴からゲート重みを算出して適用
        if self.use_gate:
            g = self.gate(torch.cat([h_pred, h_prev], dim=-1))
            out = g * out
        h_corr = h_prev + out

        # 特徴量を結合し最終融合して返す
        fused = h_corr + self.mixer(torch.cat([h_corr, h_prev], dim=-1))
        return fused


class GraphNetBlock(MessagePassing):
    """
    Graph Network (MeshGraphNet) ブロック。
    PyTorch Geometricの MessagePassing クラスをベースに、隣接格子間のメッセージ集約により
    ノード特徴量（x）とエッジ特徴量（edge_attr）を「相互に更新」するメッセージパッシングGNNのコア。
    """
    def __init__(
        self,
        hidden_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = True,
        use_rope: bool = False,
        rope_axes: int = 3,
        rope_base: float = 10000.0,
        use_gated_mlp: bool = False,
        use_gate: bool = False,
    ):
        # メッセージの集約方法は「総和 (aggr='add')」、トポロジーの流れは「始点から終点へ (source_to_target)」
        super().__init__(aggr="add", flow="source_to_target")
        
        # エッジMLPの入力は [現在のエッジ特徴量 + 始点ノード特徴量 + 終点ノード特徴量] = 3 * hidden_size
        edge_input_dim = 3 * hidden_size
        # ノードMLPの入力は [現在のノード特徴量 + 集約されたメッセージ] = 2 * hidden_size
        node_input_dim = 2 * hidden_size
        self.hidden_size = hidden_size
        self.use_gated_mlp = use_gated_mlp

        # 通常のMLPか、表現力の高いGatedMLPかをフラグによって切り替え構築
        if self.use_gated_mlp:
            self.edge_block = build_gated_mlp(in_size=edge_input_dim, hidden_size=hidden_size, out_size=hidden_size)
            self.node_block = build_gated_mlp(in_size=node_input_dim, hidden_size=hidden_size, out_size=hidden_size)
        else:
            self.edge_block = build_mlp(in_size=edge_input_dim, hidden_size=hidden_size, out_size=hidden_size,
                                        nb_of_layers=nb_of_layers, layer_norm=layer_norm)
            self.node_block = build_mlp(in_size=node_input_dim, hidden_size=hidden_size, out_size=hidden_size,
                                        nb_of_layers=nb_of_layers, layer_norm=layer_norm)

        # GNN用相対RoPE（回転位置埋め込み）の設定
        self.use_rope = use_rope
        self.rope_axes = rope_axes
        self.rope_base = rope_base

        if self.use_rope:
            if rope_axes not in (2, 3):
                raise ValueError("GNN RoPEの空間軸(rope_axes)は2または3である必要があります。")
            self._pair_count = hidden_size // (2 * rope_axes)
            self._rope_dim = self._pair_count * 2 * rope_axes
            if self._pair_count == 0:
                raise ValueError(f"hidden_size={hidden_size} が小さすぎます。RoPE回転を解くには少なくとも 2 * 空間軸 分のチャンネルが必要です。")
            
            inv = torch.arange(self._pair_count, dtype=torch.float32)
            denom = max(float(self._pair_count), 1.0)
            inv = torch.pow(self.rope_base, -inv / denom)
            self.register_buffer("_rope_inv_freq", inv, persistent=False)
        else:
            self._pair_count = 0
            self._rope_dim = 0
            self.register_buffer("_rope_inv_freq", torch.zeros(0), persistent=False)

        # メッセージ集約に対するクエリ駆動型条件付きゲートの設定
        self.use_gate = use_gate
        if self.use_gate:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.gate_pos = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,          # ノード特徴量マトリクス: [num_nodes, hidden_size]
        edge_index: torch.Tensor, # メッシュの接続情報（エッジ構造）: [2, num_edges]
        edge_attr: torch.Tensor,  # エッジ特徴量マトリクス: [num_edges, hidden_size]
        size: int = None,
        pos: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. 接続されているノードのペア（ row: 始点ノード / col: 終点ノード ）の特徴量を取得
        row, col = edge_index
        x_i = x[col]  # 終点（Target）ノード特徴量
        x_j = x[row]  # 始点（Source）ノード特徴量

        # GNN用相対RoPEが有効な場合、始点と終点の「物理的な相対距離ベクトル」に沿って特徴量を幾何回転させる
        if self.use_rope:
            if pos is None:
                raise ValueError("use_rope=True の場合、ノード座標 `pos` の提供が必須です。")
            delta_pos = pos[row, : self.rope_axes] - pos[col, : self.rope_axes] # 相対位置ベクトル
            x_j = self._apply_rope_rel(x_j, delta_pos)

        # 2. エッジブロック（MLP）を走らせ、エッジ特徴量を周辺状態を踏まえてアップデート
        edge_attr_ = self.edge_update(edge_attr, x_i, x_j)

        # 3. PyGの `propagate`（メッセージ伝播機能）を起動。内部で `message` ➔ `aggregate` ➔ `update` が順に走る
        x_ = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr_,
            size=(x.size(0), x.size(0)),
            phi=phi,
        )

        # 4. 残差結合（元の古い特徴量に、今回計算した変化量を加算）
        edge_attr = edge_attr + edge_attr_
        x = x + x_

        return x, edge_attr

    def edge_update(
        self, edge_attr: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> torch.Tensor:
        """エッジ特徴量の結合とMLP変換。"""
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        edge_attr = self.edge_block(edge_input)
        return edge_attr

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """各エッジからノードへ送信される『メッセージ』の内容。MeshGraphNetではエッジ特徴量そのものを送る。"""
        return edge_attr

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """全近傍エッジからメッセージが集約（総和）された後の、ノード特徴量の最終更新処理。"""
        
        # 条件付きマルチプリカティブゲートが有効な場合、集約されたメッセージをノード自身の状態（x）でフィルタする
        if self.use_gate:
            gate_logits = self.gate_proj(x)
            if phi is not None:
                phi = phi.view(-1, 1).to(device=gate_logits.device, dtype=gate_logits.dtype)
                gate_logits = gate_logits + phi * self.gate_pos.view(1, -1)
            gate_logits = gate_logits.to(dtype=aggr_out.dtype, device=aggr_out.device)
            gate = torch.sigmoid(gate_logits)
            aggr_out = aggr_out * gate # メッセージに重要度ゲートを乗算

        # [現在のノード特徴量 + 集約されたメッセージ] を結合してノードMLPに通す
        node_input = torch.cat([x, aggr_out], dim=-1)
        x = self.node_block(node_input)
        return x

    def _apply_rope_rel(
        self, x_src: torch.Tensor, delta_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        メッセージを送信する始点ノード（Source Node）の特徴量に対し、
        終点ノードとの空間的な相対距離（delta_pos）に応じた相対的なRoPE回転をかける内部幾何演算関数。
        """
        if self._pair_count == 0:
            return x_src

        num_edges, hidden_dim = x_src.shape
        rope_dim = self._rope_dim

        x_rot = x_src[:, :rope_dim]   # 回転を適用する周波数チャンネル領域
        x_rest = x_src[:, rope_dim:] # 回転を適用しない残りの領域

        parts = []
        start = 0
        inv_freq = self._rope_inv_freq
        delta = delta_pos.to(device=x_src.device, dtype=inv_freq.dtype)

        # X軸, Y軸, Z軸（各空間軸）ごとに独立して回転ベクトルを計算して適用（幾何構造の埋め込み）
        for axis in range(self.rope_axes):
            seg = x_rot[:, start : start + 2 * self._pair_count].reshape(num_edges, self._pair_count, 2)
            theta = delta[:, axis].unsqueeze(1) * inv_freq.unsqueeze(0) # 距離 × 周波数
            cos_theta = torch.cos(theta).to(dtype=x_src.dtype)
            sin_theta = torch.sin(theta).to(dtype=x_src.dtype)
            
            even = seg[..., 0]
            odd = seg[..., 1]
            
            # 回転行列の適用
            rot_even = even * cos_theta - odd * sin_theta
            rot_odd = even * sin_theta + odd * cos_theta
            
            seg_rot = torch.stack([rot_even, rot_odd], dim=-1).reshape(num_edges, 2 * self._pair_count)
            parts.append(seg_rot)
            start += 2 * self._pair_count

        x_rotated = torch.cat(parts, dim=-1)
        return torch.cat([x_rotated, x_rest], dim=-1)
