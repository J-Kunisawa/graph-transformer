import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch_geometric.transforms as T
from loguru import logger
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_random_edge

from graphphysics.utils.torch_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    get_masked_indexes,
)


class BaseDataset(Dataset, ABC):
    """
    基底データセットクラス（PyG Dataset を継承）

    - メタ情報は JSON (meta_path) から読み込む
    - targets: 動的に予測するフィールド名のリスト（必須）
    - preprocessing: torch_geometricTransforms の合成関数（CPU上で実行）
    - khop: k-hop を用いたグラフ拡張の度合い
    - new_edges_ratio: ランダムに追加するエッジ比率（0で無効）
    - add_edge_features: edge_attr を計算して付与するか
    - use_previous_data: データに1フレーム前の特徴を付与するか
    """

    def __init__(
        self,
        meta_path: str,
        targets: list[str],
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        world_pos_parameters: Optional[dict] = None,
    ) -> None:
        # load meta
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        # keep preprocessing on CPU; DataLoader will move Data to device
        self.device = torch.device("cpu")
        self.meta: Dict[str, Any] = meta

        # targets validation
        if not targets:
            raise ValueError("At least one target must be specified in `targets`.")
        for target in targets:
            if target not in self.meta.get("features", {}):
                raise ValueError(f"Target '{target}' not found in meta features.")
            if self.meta["features"][target].get("type") != "dynamic":
                raise ValueError(f"Target '{target}' must be a dynamic field.")
        self.targets = targets

        # trajectory information
        self.trajectory_length: int = int(self.meta["trajectory_length"])
        self.num_trajectories: Optional[int] = None

        # caches (CPU tensors)
        self.khop_edge_index_cache: Dict[int, torch.Tensor] = {}
        self.khop_edge_attr_cache: Dict[int, torch.Tensor] = {}

        # configurable behavior
        self.preprocessing = preprocessing
        self.masking_ratio = float(masking_ratio) if masking_ratio is not None else None
        self.khop = int(khop)
        self.new_edges_ratio = float(new_edges_ratio)
        self.add_edge_features = bool(add_edge_features)
        self.use_previous_data = bool(use_previous_data)

        # optional world-pos slice indices (for some transforms)
        self.world_pos_index_start: Optional[int] = None
        self.world_pos_index_end: Optional[int] = None
        if world_pos_parameters:
            self.world_pos_index_start = world_pos_parameters.get(
                "world_pos_index_start"
            )
            self.world_pos_index_end = world_pos_parameters.get("world_pos_index_end")

    @property
    @abstractmethod
    def size_dataset(self) -> int:
        """サブクラスで軌跡数を返すこと。"""

    def get_traj_frame(self, index: int) -> Tuple[int, int]:
        """
        グローバルインデックス -> (trajectory_index, frame_index)

        フレームは (trajectory_length - 1) の範囲で、target（次フレーム）が存在するように調整。
        use_previous_data が True の場合はフレームにオフセットを加える。
        """
        # If previous-frame features are requested, we must ensure that both
        # previous (frame-1) and next (frame+1) exist. Therefore the set of
        # valid "central" frames is 1 .. (T-2) inclusive (where T=trajectory_length).
        # Otherwise (no previous data), valid frames are 0 .. (T-2) so that
        # frame_target = frame + 1 stays within 0 .. (T-1).
        T = int(self.trajectory_length)
        if self.use_previous_data:
            if T < 3:
                raise ValueError("trajectory_length must be >= 3 when use_previous_data=True")
            per_traj = T - 2
            traj = index // per_traj
            frame = index % per_traj
            # shift into [1, T-2]
            frame = frame + 1
        else:
            per_traj = T - 1
            traj = index // per_traj
            frame = index % per_traj

        return traj, frame

    def __len__(self) -> int:
        # Number of valid frames per trajectory depends on whether we need
        # previous-frame features. If use_previous_data, central frames are
        # 1..T-2 (count = T-2). Otherwise frames are 0..T-2 (count = T-1).
        if self.use_previous_data:
            if int(self.trajectory_length) < 3:
                return 0
            per_traj = int(self.trajectory_length) - 2
        else:
            per_traj = int(self.trajectory_length) - 1
        return int(self.size_dataset * per_traj)

    @abstractmethod
    def __getitem__(self, index: int) -> Data:
        """サブクラスで Data を返す実装を行うこと。"""

    def _apply_preprocessing(self, graph: Data) -> Data:
        """preprocessing が与えられていれば適用して返す（CPU上）。"""
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)
        return graph

    def _add_random_edges(self, graph: Data) -> Data:
        """
        ランダムエッジを追加（new_edges_ratio に基づく）。
        edge_attr を再計算する場合は T.Cartesian / T.Distance を使う。
        """
        p = self.new_edges_ratio
        if p <= 0.0 or p > 1.0:
            return graph

        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            logger.warning("Attempted to add random edges but graph has no edge_index.")
            return graph

        new_edge_index, _ = add_random_edge(edge_index, p=p, force_undirected=True)
        graph.edge_index = new_edge_index

        if self.add_edge_features:
            # recompute edge features on CPU
            graph.edge_attr = None
            edge_feature_computer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
            graph = edge_feature_computer(graph).to(self.device)

        return graph

    def _apply_k_hop(self, graph: Data, traj_index: int) -> Data:
        """
        k-hop 拡張を適用し、可能ならキャッシュを利用する。
        キャッシュは CPU に保存し、適用時に必要な device に移動する。
        """
        if self.khop <= 1:
            return graph

        # cached?
        if traj_index in self.khop_edge_index_cache:
            cached_idx = self.khop_edge_index_cache[traj_index]
            graph.edge_index = cached_idx.to(graph.edge_index.device)
            if self.add_edge_features and traj_index in self.khop_edge_attr_cache:
                cached_attr = self.khop_edge_attr_cache[traj_index]
                if getattr(graph, "edge_attr", None) is not None and cached_attr is not None:
                    graph.edge_attr = cached_attr.to(graph.edge_attr.device)
            return graph

        # compute and cache
        if self.add_edge_features:
            khop_graph = compute_k_hop_graph(
                graph,
                num_hops=self.khop,
                add_edge_features_to_khop=True,
                device=self.device,
                world_pos_index_start=self.world_pos_index_start,
                world_pos_index_end=self.world_pos_index_end,
            )
            # cache CPU copies
            self.khop_edge_index_cache[traj_index] = khop_graph.edge_index.cpu()
            self.khop_edge_attr_cache[traj_index] = (
                khop_graph.edge_attr.cpu() if getattr(khop_graph, "edge_attr", None) is not None else None
            )
            # apply to current graph (move to its device)
            graph.edge_index = khop_graph.edge_index.to(graph.edge_index.device)
            if getattr(khop_graph, "edge_attr", None) is not None and getattr(graph, "edge_attr", None) is not None:
                graph.edge_attr = khop_graph.edge_attr.to(graph.edge_attr.device)
        else:
            khop_idx = compute_k_hop_edge_index(graph.edge_index, self.khop, graph.num_nodes)
            self.khop_edge_index_cache[traj_index] = khop_idx.cpu()
            graph.edge_index = khop_idx.to(graph.edge_index.device)

        return graph

    def _may_remove_edges_attr(self, graph: Data) -> Data:
        """add_edge_features が False のとき edge_attr を削除する。"""
        if not self.add_edge_features:
            graph.edge_attr = None
        return graph

    def _get_masked_indexes(self, graph: Data) -> Optional[torch.Tensor]:
        """masking_ratio が設定されていれば選択インデックスを返す。"""
        if self.masking_ratio is None:
            return None
        return get_masked_indexes(graph, self.masking_ratio)