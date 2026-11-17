from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.transforms import KNNGraph


class UpSampler(nn.Module):
    """
    アップサンプラー（逆プーリング層）。
    粗いメッシュ（コース格子）上で計算された潜在特徴量を、k近傍内挿法（k-NN補間）を用いて
    元の高密度なメッシュ（ファイン格子）の座標へとマッピングして戻す。
    """
    def __init__(self, d_in: int, d_out: int, k: int = 6):
        super().__init__()
        self.k = k  # 補間時に参照する近傍ノード数
        self.lin = nn.Linear(d_in, d_out)  # 補間後の特徴量を線形変換する全結合層

    @torch.compiler.disable  # PyTorchのコンパイルによるグラフ不整合を防ぐためのデコレータ
    def forward(
        self,
        x_coarse: torch.Tensor,    # コース格子のノード特徴量: [C, d_in] (C: コースノード数)
        pos_coarse: torch.Tensor,  # コース格子の3次元座標: [C, pos_dim]
        pos_fine: torch.Tensor,    # 元のファイン格子の3次元座標: [N, pos_dim] (N: ファインノード数)
        batch_coarse: Optional[torch.Tensor] = None, # コース格子のミニバッチ割当
        batch_fine: Optional[torch.Tensor] = None,   # ファイン格子のミニバッチ割当
    ) -> torch.Tensor:
        
        # 3次元空間上の距離の逆数を重みとして、コース格子の特徴量をファイン格子の座標へ補間展開
        interp = knn_interpolate(
            x=x_coarse,
            pos_x=pos_coarse,
            pos_y=pos_fine,
            batch_x=batch_coarse,
            batch_y=batch_fine,
            k=self.k,
        )  # 補間後のテンソル形状: [N, d_in]
        
        # 特徴次元を目的の次元 (d_out) へ写像して返す
        return self.lin(interp)


class DownSampler(nn.Module):
    """
    ダウンサンプラー（プーリング層）。
    高密度なグラフから重要なノードをトポロジーを考慮しつつ間引き、
    ノード数が削減された粗いグラフ（コース格子）を動的に再構築する。
    """
    def __init__(self, d_in: int, d_out: int, ratio: int = 0.25):
        super().__init__()

        self.ratio = ratio  # サンプリングして残すノードの割合（例: 0.25 = 25%のノードを維持）
        self.lin = nn.Linear(d_in, d_out)  # プーリングと同時に特徴量を変換する全結合層
        self.min_score = None
        self.nonlinearity = "softmax"

        # 特徴量（またはアテンション重み）に基づいてスコア上位のノードを抽出するモジュール
        self.select = SelectTopK(d_in, self.ratio, self.min_score, self.nonlinearity)
        
        # 間引かれたノード同士を空間的な近接度（k=6）ベースで繋ぎ直し、新しいエッジ（トポロジー）を再構築するクラス
        self.remesher = KNNGraph(k=6, force_undirected=True)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,      # ファイン格子の特徴量テンソル
        pos: torch.Tensor,    # ファイン格子の3次元座標テンソル
        batch: torch.Tensor,  # ミニバッチの割当インデックス
        attn: Optional[torch.Tensor] = None, # スコア計算に用いるテンソル（未指定時は特徴量 x をそのまま使用）
    ) -> Batch:

        # サンプリングの評価基準となるテンソルを決定
        attn = x if attn is None else attn
        
        # Top-Kアルゴリズムにより、残すノードのインデックス（perm）を選出
        select_out = self.select(attn, batch)
        perm = select_out.node_index  # 維持されるノードのインデックス配列

        # 選ばれたノードの情報のみを抽出してダウンサンプリング
        x_c = self.lin(x[perm])  # 特徴量を抽出しつつ次元変換
        pos_c = pos[perm]        # 座標の抽出
        batch_c = batch[perm]    # バッチ情報の抽出

        # 抽出したノード情報で仮のグラフ（エッジが無い状態）を構築
        coarse_graph = Batch(
            x=x_c,
            batch=batch_c,
            pos=pos_c,
        )

        # KNNGraphを適用して、空間的に近いノード間に新しいエッジ（edge_index）を自動生成して返す
        return self.remesher(coarse_graph)