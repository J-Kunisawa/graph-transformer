from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from graphphysics.models.layers import RMSNorm, build_gated_mlp

# --- 内部補助コンポーネント (Transformerブロック) ---

class _EncoderBlock(nn.Module):
    """
    1つのTransformerエンコーダブロック。
    星型部分グラフ（パックされた中心と近傍）の内部で、相互のアテンション相互作用を計算する。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = RMSNorm(d_model)
        self.ffn = build_gated_mlp(
            in_size=d_model, hidden_size=d_model, out_size=d_model
        )

    def forward(self, x, key_padding_mask=None):
        # プリノルマライズ（Pre-LN）スタイルを採用
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask, # パディング（-1埋め）領域をアテンションから除外
            need_weights=False,
        )
        x = x + attn_out # 残差結合
        x = x + self.ffn(self.ln2(x)) # Gated-MLPによる特徴量変換と残差結合
        return x


class _Encoder(nn.Module):
    """上記ブロックを複数層スタックするためのエンコーダラッパー。"""
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_EncoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, key_padding_mask=None):
        for lyr in self.layers:
            x = lyr(x, key_padding_mask=key_padding_mask)
        return x


# --- 幾何・グラフ構造操作用ヘルパー関数 ---

def _make_undirected(edge_index: torch.Tensor, assume_undirected: bool) -> torch.Tensor:
    """
    グラフのエッジ配列に対し、逆方向のエッジを結合して無向グラフ化する。
    重複排除（unique）を行わないことで最速で処理を完了させる。
    """
    if assume_undirected:
        return edge_index.long()
    e = edge_index.long()
    rev = torch.stack([e[1], e[0]], dim=0) # 始点と終点を反転
    return torch.cat([e, rev], dim=1) # 元のエッジに結合


def _sorted_by_src(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    エッジを始点（src）ノードIDでソートし、CSR（Compressed Sparse Row）風の構造を構築する。
    
    戻り値:
        - src_s: ソートされた始点ノードID配列
        - dst_s: 対応する終点（隣接）ノードID配列
        - row_ptr: 各ノードの隣接データがdst_sのどこから始まっているかを示すポインタ [N + 1]
    """
    e = edge_index.long()
    src, dst = e[0], e[1]
    order = torch.argsort(src) # 始点ノードIDで並び替えるためのインデックスを取得
    src_s = src[order]
    dst_s = dst[order]

    # 各ノードが何個の隣接（エッジ）を持っているかをカウント
    counts = torch.bincount(src_s, minlength=num_nodes)  # [N]
    
    # カウントの累積和をとることで、CSRポインタポインタ（row_ptr）を作成
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=e.device)
    row_ptr[1:] = counts.cumsum(0)
    return src_s, dst_s, row_ptr


class SpatialMTP1Hop(nn.Module):
    """
    空間ターゲット予測 (Spatial Masked Target Prediction) レイヤー。
    
    中心ノード（centers）に隣接する1ホップのローカル領域を「星型（Star）」として
    一括ベクトルパッキングし、近傍の特徴量から中心のターゲット物理量を予測する補助タスクを行う。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 1,
        assume_undirected: bool = True,
        max_neighbors: Optional[int] = None,
    ):
        super().__init__()
        self.enc = _Encoder(d_model, num_heads, num_layers)
        self.in_ln = RMSNorm(d_model)
        self.assume_undirected = assume_undirected
        self.max_neighbors = max_neighbors # 1ノードあたりの最大サンプリング隣接数（メモリ制限用）

    @torch.no_grad()
    def _cap_neighbors(
        self, dst_s: torch.Tensor, row_ptr: torch.Tensor, centers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        選ばれた中心ノード（centers）たちの隣接リスト（dst_s）を切り出し、
        max_neighbors を超えている場合は一様ランダムサンプリングによって制限をかける。
        """
        device = dst_s.device
        B = centers.numel()
        starts = row_ptr[centers]
        ends = row_ptr[centers + 1]
        counts = ends - starts  # 各中心ノードが持つ本来の隣接（次数）数: [B]
        
        # 1. 上限（max_neighbors）設定がない場合は、そのままスライスしてフラットに結合
        if self.max_neighbors is None:
            total = int(counts.sum().item())
            out = dst_s.new_empty(total)
            offset = 0
            for b in range(B):
                s, e = int(starts[b].item()), int(ends[b].item())
                if e > s:
                    out[offset : offset + (e - s)] = dst_s[s:e]
                    offset += e - s
            return out, counts

        # 2. 上限がある場合：上限を超える行に対して、非復元抽出によるランダムサンプリングを実行
        k = self.max_neighbors
        kept_counts = torch.clamp(counts, max=k) # 各中心がキープする実際の数
        total = int(kept_counts.sum().item())
        out = dst_s.new_empty(total)
        offset = 0
        for b in range(B):
            s, e = int(starts[b].item()), int(ends[b].item())
            c = int(counts[b].item())
            kk = min(k, c)
            if kk > 0:
                if c == kk:
                    out[offset : offset + kk] = dst_s[s:e]
                else:
                    # インデックスのランダムシャッフルから上位kk個を切り出し（非復元抽出）
                    perm = torch.randperm(c, device=device)[:kk]
                    out[offset : offset + kk] = dst_s[s + perm]
                offset += kk
        return out, kept_counts

    def forward(
        self,
        H: torch.Tensor,           # 中心ノード用特徴量マトリクス: [N, d_model]
        edge_index: torch.Tensor,  # グラフのエッジ構造: [2, M]
        centers: torch.Tensor,     # 今回MTPタスクのターゲットとして評価する中心ノードID群: [B]
        out_head: nn.Module,       # 予測値を実物理量にマップする共通の出力ヘッド(MLP): d_model -> y_dim
        target: torch.Tensor,      # 正解（グラウンドトゥルース）物理量マトリクス: [N, y_dim]
        reduction: str = "mean_per_center",
        row_ptr: torch.Tensor | None = None,
        dst_sorted: torch.Tensor | None = None,
        H_neigh: torch.Tensor | None = None, # 隣接ノード用特徴量。未指定時は H をそのまま使用
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = H.device
        N, d = H.shape
        centers = centers.to(device=device, dtype=torch.long)

        # ターゲットセンターが空の場合は速やかにLoss 0を返す
        if centers.numel() == 0:
            zero = H.new_tensor(0.0)
            return zero, {"sp_mtp/centers": H.new_tensor(0), "sp_mtp/pairs": H.new_tensor(0)}

        # CSR隣接構造の準備（外部からキャッシュが渡されていれば再利用、無ければ初回ビルド）
        if (row_ptr is not None) and (dst_sorted is not None):
            row_ptr = row_ptr.to(device)
            dst_s = dst_sorted.to(device)
        else:
            e = _make_undirected(edge_index.to(device), assume_undirected=self.assume_undirected)
            _, dst_s, row_ptr = _sorted_by_src(e, num_nodes=N)

        # 指定された中心ノードに対応する隣接リストを切り出し（およびサンプリング上限処理）
        dst_concat, counts = self._cap_neighbors(dst_s, row_ptr, centers)  # フラット配列 [T], カウント [B]
        B = centers.numel()
        deg = counts  # 各中心ノードの確定近傍数
        total_neighbors = int(deg.sum().item())
        
        if total_neighbors == 0:
            zero = H.new_tensor(0.0)
            return zero, {"sp_mtp/centers": torch.tensor(float(B), device=device), "sp_mtp/pairs": H.new_tensor(0)}

        # --- ここからスター・パッキング（Vectorized Star Packing）のコアロジック ---
        # ループを一切回さずに、各中心とその隣接ノードを格納する固定長（L = 1 + max_deg）のテンソルを一括作成する
        max_deg = int(deg.max().item())
        L = 1 + max_deg # シークエンス長（ 1(中心) + 最大隣接数 ）
        idx_mat = torch.full((B, L), -1, dtype=torch.long, device=device) # パディング値 -1 で初期化
        idx_mat[:, 0] = centers # 最初の列（インデックス0）には中心ノードのIDをセット

        # ブロードキャスト用マスクの作成
        pos = torch.arange(max_deg, device=device).unsqueeze(0)  # [1, max_deg]
        mask = pos < deg.unsqueeze(1)  # [B, max_deg] (実際の隣接数が存在する領域だけがTrueになる)

        # フラット配列（dst_concat）からピンポイントでデータを引き抜くためのインデックスマトリクスを計算
        pref = torch.zeros_like(deg)
        pref[1:] = deg.cumsum(0)[:-1]
        idx_in_concat = pref.unsqueeze(1) + pos  # [B, max_deg]

        # マスクを適用して、パッキング行列の1列目以降（近傍領域）にノードIDを射影
        nbrs_full = torch.empty(B, max_deg, dtype=torch.long, device=device)
        nbrs_full[mask] = dst_concat[idx_in_concat[mask]]
        idx_mat[:, 1:][mask] = nbrs_full[mask]

        # パディング（-1）の場所を示すアテンションマスク
        key_padding_mask = idx_mat.eq(-1)

        # --- 入力特徴量テンソル（X）の組み上げ ---
        safe_idx = idx_mat.clamp_min(0)  # パディング領域(-1)でのインデックスエラーを防ぐダミー処置
        X = torch.zeros(B, L, d, device=device, dtype=H.dtype)
        
        # 0列目（中心）には、中心用ノード特徴量（H）をセット
        X[:, 0] = H[safe_idx[:, 0]]

        # 1列目以降（近傍）には、近傍用ノード特徴量（H_neigh または H）をセット
        neigh_src = H if H_neigh is None else H_neigh
        if max_deg > 0:
            X[:, 1:][mask] = neigh_src[nbrs_full[mask]]

        # レイヤーノーマライズをかけ、パディング領域の特徴量を完全に0.0に消去
        X = self.in_ln(X)
        X[key_padding_mask] = 0.0

        # --- Transformerエンコーダによる相互作用演算 ---
        # 星型部分グラフのシークエンス内でアテンションを回す
        Z = self.enc(X, key_padding_mask=key_padding_mask)  # 出力形状: [B, L, d]

        # --- 誤差の計算とロス削減処理 ---
        # 予測対象となるフロンティア（近傍から見た中心の情報、またはその逆の予測領域）のテンソルを切り出す
        owners = torch.repeat_interleave(torch.arange(B, device=device), deg)  # 各隣接がどのバッチ（中心）に属するか: [T]
        targets = idx_mat[:, 1:][mask]  # 正解ノードのID群 [T]
        Z_frontier = Z[:, 1:][mask]  # 対応する潜在空間出力特徴量 [T, d]

        y_hat = out_head(Z_frontier)  # 出力ヘッドに通して物理量を予測 [T, y_dim]
        y_true = target[targets]      # 正解の物理量 [T, y_dim]
        err = (y_hat - y_true).pow(2).mean(dim=-1)  # 各ペアごとのL2誤差 [T]

        # 設定された縮小（reduction）方法に従って最終的なAuxiliary Lossを算出
        if reduction == "mean":
            aux_loss = err.mean() # 全ペアの一律平均
        else:
            # "mean_per_center": 中心ノード（スター）ごとに一度平均を出してから、バッチ全体で平均化する（次数による不平等を無くす）
            loss_sum = torch.zeros(B, device=device, dtype=err.dtype).scatter_add_(
                0, owners, err
            )
            cnt = deg.to(err.dtype).clamp_min(1.0)
            aux_loss = (loss_sum / cnt).mean()

        # 進捗ログ（WandBやコンソール用）の統計情報をまとめて返す
        stats = {
            "sp_mtp/centers": torch.tensor(float(B), device=device),
            "sp_mtp/pairs": torch.tensor(float(total_neighbors), device=device),
            "sp_mtp/mean_pair_loss": err.mean().detach(),
            "sp_mtp/max_deg": torch.tensor(float(max_deg), device=device),
        }
        return aux_loss, stats