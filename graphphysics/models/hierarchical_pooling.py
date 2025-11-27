from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from torch_geometric.data import Batch, Data
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.transforms import KNNGraph


class UpSampler(nn.Module):
    """
    点集合を細分化（補間）して特徴次元を変換するモジュール。

    Args:
        d_in: 入力特徴次元
        d_out: 出力特徴次元
        k: knn_interpolate の近傍数
    """

    def __init__(self, d_in: int, d_out: int, k: int = 6) -> None:
        super().__init__()
        self.k = int(k)
        self.lin = nn.Linear(int(d_in), int(d_out))

    @torch.compiler.disable
    def forward(
        self,
        x_coarse: torch.Tensor,  # [C, d_in]
        pos_coarse: torch.Tensor,  # [C, pos_dim]
        pos_fine: torch.Tensor,  # [N, pos_dim]
        batch_coarse: Optional[torch.Tensor] = None,
        batch_fine: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Coarse->Fine に knn 補間を実行して線形変換を適用する。
        入力の形状チェックを行い、問題があれば例外を投げる。
        """
        if x_coarse is None or pos_coarse is None or pos_fine is None:
            raise ValueError("x_coarse, pos_coarse and pos_fine must be provided")

        interp = knn_interpolate(
            x=x_coarse,
            pos_x=pos_coarse,
            pos_y=pos_fine,
            batch_x=batch_coarse,
            batch_y=batch_fine,
            k=self.k,
        )  # [N, d_in]
        return self.lin(interp)


class DownSampler(nn.Module):
    """
    グラフのダウンサンプリング（Top-K 選択 + 再メッシュ）を行うモジュール。

    Args:
        d_in: 入力特徴次元（SelectTopK に渡す）
        d_out: ダウンサンプリング後の特徴次元（線形変換で出力）
        ratio: 取り残すノードの割合（0.0 < ratio <= 1.0）
    """

    def __init__(self, d_in: int, d_out: int, ratio: float = 0.25) -> None:
        super().__init__()

        if not (0.0 < float(ratio) <= 1.0):
            raise ValueError("ratio must be in (0, 1]")

        self.ratio = float(ratio)
        self.lin = nn.Linear(int(d_in), int(d_out))
        self.min_score = None
        self.nonlinearity = "softmax"

        # SelectTopK: in_channels, ratio, min_score, nonlinearity
        self.select = SelectTopK(int(d_in), self.ratio, self.min_score, self.nonlinearity)
        # remesher: KNNGraph transform を用いて coarse graph の近傍構造を再構成
        self.remesher = KNNGraph(k=6, force_undirected=True)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> Data:
        """
        x, pos, batch を受け取り top-k 選択を行い、その結果を線形変換して再メッシュする。
        戻り値は remesher が返す Data（粗解像度グラフ）。
        """
        if x is None or pos is None or batch is None:
            raise ValueError("x, pos and batch tensors must be provided")

        # attention スコアとして x をデフォルトで使用
        attn_tensor = x if attn is None else attn
        if attn_tensor.dim() == 1:
            attn_tensor = attn_tensor.unsqueeze(1)

        # SelectTopK により選択インデックスを取得
        select_out = self.select(attn_tensor, batch)
        # 選択ノードのパーミッション（インデックス）
        perm = getattr(select_out, "node_index", None)
        if perm is None:
            # 互換性のため node_idx / idx の可能性もチェック
            perm = getattr(select_out, "idx", None)
        if perm is None:
            raise RuntimeError("SelectTopK did not produce node indices")

        perm = perm.long()

        # 粗ノードの特徴・位置・バッチを作成し線形変換を適用
        x_c = self.lin(x[perm])
        pos_c = pos[perm]
        batch_c = batch[perm]

        coarse_graph = Batch(x=x_c, batch=batch_c, pos=pos_c)

        # remesher (KNNGraph) を適用して粗グラフを返す
        remeshed = self.remesher(coarse_graph)
        return remeshed