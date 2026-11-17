# --------------------------------------------------------------------------------------
# 免責事項:
# このファイルは TransolverPlusPlus の GitHub リポジトリから翻案・拡張されたものです。
# https://github.com/thuml/Transolver_plus/tree/main
# 2025年9月17日取得。
# --------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from graphphysics.models.layers import TemporalAttention, build_mlp


def gumbel_softmax(logits, tau=1, hard=False):
    """
    ガンベル・ソフトマックス（Gumbel-Softmax）サンプリング。
    確率分布（logits）に対して微分可能性を保ったまま、カテゴリカルなサンプリング（離散的な選択）を近似する。
    Transolverにおいては、各ノードをどの「空間スライス領域（トークン）」に割り当てるかを動的に決定するのに使用される。
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau # 温度パラメータ（tau）によるスケーリング

    y = F.softmax(y, dim=-1)

    # hard=True の場合、順伝播では完全に1値（One-Hot）化し、逆伝播ではStraight-Through Estimatorにより微分可能にする
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_1D_Eidetic(nn.Module):
    """
    物理特化型スライス・アテンション・メカニズム。
    メッシュ全体の膨大なノード情報を `slice_num` 個の代表トークンへ一度集約（圧縮）し、
    その代表トークン間で自己アテンションを解いたのち、再び元の全ノードへ情報を復元（逆射影）する。
    これにより、計算コストを O(N^2) から O(slice_num^2) へと劇的に削減する。
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        use_rope_embeddings: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # 空間スライスを学習するためのバイアスと射影ネットワーク
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num), nn.GELU(),
            nn.Linear(slice_num, 1), nn.GELU(),
        )
        
        # 空間位置埋め込み（RoPE）の拡張設定
        self.use_rope_embeddings = use_rope_embeddings
        self.rope_pos_dimension = rope_pos_dimension
        self.rope_base = rope_base
        self.use_gated_attention = use_gated_attention
        
        if self.use_rope_embeddings:
            if rope_pos_dimension <= 0:
                raise ValueError("use_rope_embeddings=True の場合、rope_pos_dimension は正の整数である必要があります。")
            self.rope_projection = nn.Linear(rope_pos_dimension * 2, dim_head)
        else:
            self.rope_projection = None

        # 特徴量の線形写像層
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        # 直交初期化を適用して初期のスライス分離能力を高める
        for ps in [self.in_project_slice]:
            torch.nn.init.orthogonal_(ps.weight)
            
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        
        # 条件付きアテンションゲートの拡張
        if self.use_gated_attention:
            self.attn_gate = nn.Sequential(
                nn.Linear(2 * dim_head, dim_head), nn.SiLU(),
                nn.Linear(dim_head, dim_head), nn.Sigmoid(),
            )
        else:
            self.attn_gate = None

    def _rope_features(self, pos: torch.Tensor) -> torch.Tensor:
        """3次元等の空間座標（pos）から、RoPE用のサイン・コサイン角度場特徴量を計算する。"""
        if self.rope_pos_dimension <= 0:
            return torch.zeros(pos.shape[0], pos.shape[1], 0, device=pos.device, dtype=pos.dtype)
        if pos.size(-1) < self.rope_pos_dimension:
            raise ValueError(f"要求されたRoPE軸数 {self.rope_pos_dimension} に対し、入力の座標次元数が {pos.size(-1)} と不足しています。")
            
        pos_slice = pos[..., : self.rope_pos_dimension]
        log_base = torch.log(torch.tensor(self.rope_base, device=pos.device, dtype=pos.dtype))
        inv_freq = torch.exp(-torch.arange(self.rope_pos_dimension, device=pos.device, dtype=pos.dtype) * log_base / max(self.rope_pos_dimension, 1))
        angles = pos_slice * inv_freq.view(1, 1, self.rope_pos_dimension)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(self, x, pos=None):
        B, N, C = x.shape # B: バッチサイズ, N: メッシュのノード数, C: 特徴次元数

        # 1. 特徴量をマルチヘッドテンソルへ射影
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # 形状: [B, Heads, NumNodes, DimHead]
        
        # 幾何位置情報（RoPE）が有効な場合、特徴量に直接位置バイアスを加算結合
        if self.use_rope_embeddings and pos is not None:
            rope_feat = self._rope_features(pos).to(x_mid.dtype)
            rope_proj = self.rope_projection(rope_feat)
            x_mid = x_mid + rope_proj.unsqueeze(1)

        # 2. ガンベル・ソフトマックスによる、各ノードの「空間スライス領域」への割り当て重みを算出
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01) # 数値安定化のためのクリップ
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature) # [B, Heads, NumNodes, SliceNum]
        
        # マルチGPUによる分散並列学習時（DDP）のための、スライス統計情報の同期処理
        slice_norm = slice_weights.sum(2)  # [B, Heads, SliceNum]
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM)
            
        # 3. ノード特徴量をスライス重みを用いてアインシュタイン縮約積により「集約（圧縮）」し、固定長のスライストークンを生成
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous() # [B, Heads, SliceNum, DimHead]
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # 4. 集約されたスライストークン空間（SliceNum）の中で、標準的な自己アテンションを高速実行
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        ) # 出力形状: [B, Heads, SliceNum, DimHead]
        
        # 条件付きゲートによるアテンション出力のフィルタリング
        if self.attn_gate is not None:
            gate_input = torch.cat([slice_token, out_slice_token], dim=-1)
            gate = self.attn_gate(gate_input)
            out_slice_token = gate * out_slice_token

        # 5. アテンションを適用したスライストークンから、元の全ノード（NumNodes）の位置へと情報を逆マッピング（再展開）
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)") # ヘッドの結合
        return self.to_out(out_x)


class Transolver_plus_block(nn.Module):
    """
    Transolver++ ブロック。
    [ 正規化 ➔ スライスアテンション ➔ 残差結合 ] ➔ [ 正規化 ➔ MLP（build_mlpによる特徴変換） ➔ 残差結合 ]
    という統合マクロスタックレイヤー。VRAM節約のためのアクティベーション・チェックポイントを内包。
    """
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_rope_embeddings: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, use_rope_embeddings=use_rope_embeddings,
            rope_pos_dimension=rope_pos_dimension, rope_base=rope_base, use_gated_attention=use_gated_attention,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        # 共有レイヤー部品（build_mlp）を利用したFFN（Feed-Forward）の定義
        self.mlp = build_mlp(
            in_size=hidden_dim, hidden_size=hidden_dim * mlp_ratio, out_size=hidden_dim,
            nb_of_layers=2, layer_norm=False, act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, pos=None):
        # 訓練時かつ勾配計算有効時は、チェックポイント（再計算）をかけてVRAM消費量を極限まで抑える
        if self.training:
            fx = checkpoint(lambda z: self.Attn(z, pos), self.ln_1(fx), use_reentrant=True) + fx
        else:
            fx += self.Attn(self.ln_1(fx), pos)
            
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
            
        # 最終層の設定がONの場合は、出力をターゲット次元（out_dim）へマッピング
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    """
    Transolver++ ネットワーク本体。
    不規則メッシュの形状の壁を打ち破り、あらゆる三次元偏微分方程式フィールドを統一的に解くための基盤モデル。
    """
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        super(Model, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref # 統一参照グリッドの1軸あたりの解像度
        self.unified_pos = unified_pos # 統一幾何グリッド埋め込みを有効にするかのフラグ
        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.rope_pos_dimension = rope_pos_dimension
        self.rope_base = rope_base
        self.use_temporal_block = use_temporal_block
        
        # 統一幾何グリッド（unified_pos=True）の場合、入力特徴量サイズに基準格子（ref^3）への距離フィールドが追加される
        if self.unified_pos:
            self.preprocess = build_mlp(
                in_size=fun_dim + self.ref * self.ref * self.ref, hidden_size=n_hidden * 2,
                out_size=n_hidden, nb_of_layers=2, layer_norm=False, act=act,
            )
        else:
            self.preprocess = build_mlp(
                in_size=fun_dim + space_dim, hidden_size=n_hidden * 2,
                out_size=n_hidden, nb_of_layers=2, layer_norm=False, act=act,
            )

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(3, n_hidden) # グローバル条件ベクトル用の線形埋め込み
        
        # 共通引数の辞書化
        block_kwargs = dict(
            num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
            mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
            use_rope_embeddings=use_rope_embeddings, rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base, use_gated_attention=use_gated_attention,
        )
        
        # 時間補正アテンションの有無によって、プロセッサブロックの出力マッピングの終端方法を変更
        if self.use_temporal_block:
            self.blocks = nn.ModuleList([Transolver_plus_block(**block_kwargs, last_layer=False) for _ in range(n_layers)])
            self.output_proj = nn.Linear(n_hidden, out_dim)
        else:
            # 時間補正が無い標準スタイル：最後のレイヤー（_ == n_layers - 1）でのみ、内部で自動マッピングさせる
            self.blocks = nn.ModuleList([Transolver_plus_block(**block_kwargs, last_layer=(_ == n_layers - 1)) for _ in range(n_layers)])
            self.output_proj = None
            
        # ゼロ埋め初期特徴量の偏りを防ぐための学習可能なプレースホルダーパラメータ
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        
        # 時間発展の物理的一貫性を維持するクロスアテンション修正レイヤーの拡張配置
        self.temporal_block = (
            TemporalAttention(hidden_size=n_hidden, num_heads=n_head, use_gate=use_gated_attention)
            if self.use_temporal_block
            else None
        )

    def get_grid(self, my_pos):
        """
        統一位置情報用（Unified Position）幾何演算コア。
        3次元空間上に仮想の等間隔直交グリッド（ref^3個の基準点）を生成し、
        入力された不規則メッシュ座標（my_pos）から各基準格子へのユークリッド距離マトリクスを総当たり計算する。
        """
        batchsize = my_pos.shape[0]

        # X, Y, Zの3軸の直交参照格子空間を生成
        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        
        # 結合して [Batch, ref^3, 3] の絶対座標格子を構築
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref**3, 3)

        # ブロードキャスト計算：入力座標と仮想格子の全ペアの2乗和平方根（ユークリッド距離場）を一発算出
        pos = (
            torch.sqrt(torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1))
            .reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos # 形状: [Batch, NumNodes, ref^3]

    def forward(self, x, pos, condition):
        if self.use_rope_embeddings and pos is None:
            raise ValueError("use_rope_embeddings=True の場合は Transolver にノード位置座標（pos）を渡す必要があります。")
            
        fx = None
        
        # 統一幾何グリッドが有効な場合、特徴量ベクトル（x）の末尾に仮想格子への距離場をガッチャンコする
        if pos is not None and self.unified_pos:
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)

        # 特徴量の前置エンコード（preprocess）とプレースホルダーバイアスの加算
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        # グローバル条件ベクトル（レイノルズ数やマッハ数、マクロ境界情報等）がある場合は、埋め込んで各ノードにブロードキャスト加算
        if condition is not None:
            condition = self.embedding(condition).unsqueeze(1)
            fx = fx + condition

        # レイヤーブロック（自己アテンション）を順次実行
        prev_fx = fx
        for block in self.blocks:
            prev_fx = fx
            fx = block(fx, pos=pos)
            
        # 時間発展補正ブロック（Temporal Block）が有効な場合、バッチ内の各データごとにクロスアテンションを適用
        if self.temporal_block is not None:
            fused = []
            for b in range(fx.size(0)):
                fused.append(self.temporal_block(prev_fx[b], fx[b], None))
            fx = torch.stack(fused, dim=0)
            fx = self.output_proj(fx) # 最終次元へ線形マッピング
        elif self.output_proj is not None:
            fx = self.output_proj(fx)
            
        return fx