from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
import torch
from loguru import logger
from torch.utils.data import get_worker_info
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.hierarchical import (
    get_frame_as_graph,
    get_traj_as_meshes,
    read_h5_metadata,
)

try:
    import h5py
except ImportError as exc:
    raise RuntimeError("h5py is required to use H5Dataset") from exc


class H5Dataset(BaseDataset):
    """
    HDF5 ベースのデータセット。

    特徴:
    - worker ごとに h5py.File ハンドルを保持（get_worker_info を利用）
    - 軌跡・フレームごとの変換結果をキャッシュして再計算を避ける
    - add_edge_features / khop / masking など BaseDataset の設定を尊重
    """

    def __init__(
        self,
        h5_path: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        world_pos_parameters: Optional[dict] = None,
        cache_size: int = 8,
    ):
        # BaseDataset は meta_path を読み込み self.meta を初期化する
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            world_pos_parameters=world_pos_parameters,
        )

        self.type = "h5"

        # train <-> test 切替（validation 用スイッチ）
        if switch_to_val:
            h5_path = h5_path.replace("train", "test")

        self.h5_path = h5_path
        self.meta_path = meta_path
        self.cache_size = int(cache_size)

        # metadata と trajectory index を読み取り（より完全な meta を上書き）
        (
            self.datasets_index,
            self._size_dataset,
            file_meta,
        ) = read_h5_metadata(dataset_path=h5_path, meta_path=meta_path)
        # read_h5_metadata が返す meta を採用（BaseDataset の meta と差分があればこちらが正）
        if file_meta is not None:
            self.meta = file_meta

        # timestep (dt) のフォールバックと警告処理
        self.dt = float(self.meta.get("dt", 1.0))
        if self.dt == 0:
            self.dt = 1.0
            logger.warning(
                "The dataset has dt == 0 in meta; falling back to dt=1.0 to allow XDMF export."
            )

        # 内部キャッシュ / ハンドル
        self._file_handles: Dict[str, h5py.File] = {}
        self._trajectory_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._frame_cache: "OrderedDict[Tuple[str, int], Tuple[Data, Optional[torch.Tensor]]]" = OrderedDict()

    @property
    def size_dataset(self) -> int:
        """データセット中の軌跡数を返す。"""
        return int(self._size_dataset)

    def _close_file_handles(self) -> None:
        """全ワーカーの h5py.File ハンドルを安全にクローズする。"""
        for handle in list(self._file_handles.values()):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                # 破棄時の例外は無視して続行
                pass
        self._file_handles.clear()

    def _get_file_handle(self) -> h5py.File:
        """
        ワーカーごとの h5py.File を返す。worker ごとに独立したハンドルを持つことで
        マルチプロセス読み出し時の競合を避ける。
        """
        worker_info = get_worker_info()
        worker_id = str(worker_info.id) if worker_info is not None else "main"
        handle = self._file_handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._file_handles[worker_id] = handle
        return handle

    def _get_trajectory(self, traj_number: str) -> dict:
        """traj_number に対応する軌跡データをキャッシュ付きで取得する。"""
        cached = self._trajectory_cache.get(traj_number)
        if cached is not None:
            # LRU 効果のため末尾に移動
            self._trajectory_cache.move_to_end(traj_number)
            return cached

        fh = self._get_file_handle()
        trajectory = get_traj_as_meshes(file_handle=fh, traj_number=traj_number, meta=self.meta)
        self._trajectory_cache[traj_number] = trajectory
        if len(self._trajectory_cache) > self.cache_size:
            self._trajectory_cache.popitem(last=False)
        return trajectory

    def _cache_graph(
        self,
        key: tuple[str, int],
        graph: Data,
        selected_indices: Optional[torch.Tensor],
    ) -> None:
        """(graph, selected_indices) をフレームキャッシュに保存する。"""
        self._frame_cache[key] = (graph, selected_indices)
        self._frame_cache.move_to_end(key)
        if len(self._frame_cache) > self.cache_size * 2:
            self._frame_cache.popitem(last=False)

    def _build_node_features(self, traj: dict, frame: int) -> torch.Tensor:
        """
        軌跡 dict からノード特徴行列を作る（time 列を末尾に追加）。
        戻り値は torch.float32 のテンソル。
        """
        time = frame * float(self.meta.get("dt", 1.0))

        point_data = {
            key: (traj[key][frame] if traj[key].ndim > 1 else traj[key])
            for key in traj.keys()
            if key not in ("mesh_pos", "cells", "node_type")
        }
        # node_type は通常 shape=(N,) もしくは (1,N) で保存されるので整形
        point_data["node_type"] = traj["node_type"][0]

        arrays = []
        for arr in point_data.values():
            a = arr
            if a.ndim == 1:
                a = a[:, None]
            arrays.append(a.astype(np.float32))

        if arrays:
            node_features = np.concatenate(arrays, axis=1)
        else:
            node_features = np.zeros((traj["mesh_pos"].shape[-2], 0), dtype=np.float32)

        time_column = np.full((node_features.shape[0], 1), time, dtype=np.float32)
        node_features = np.concatenate([node_features, time_column], axis=1)
        return torch.from_numpy(node_features)

    def _get_processed_graph(
        self,
        traj_index: int,
        frame: int,
        traj: Optional[dict] = None,
    ) -> Tuple[Data, Optional[torch.Tensor]]:
        """
        指定 traj_index, frame に対応する加工済みグラフを返す（キャッシュ使用）。
        返り値は (graph, selected_indices) — masking がなければ selected_indices は None。
        """
        traj_number = self.datasets_index[traj_index]
        cache_key = (traj_number, frame)

        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            graph, selected_indices = cached
        else:
            if traj is None:
                traj = self._get_trajectory(traj_number)

            graph = get_frame_as_graph(
                traj=traj,
                frame=frame,
                meta=self.meta,
                targets=self.targets,
                frame_target=frame + 1,
            )

            # preprocessing / k-hop / edge features / random edges / masking の順で適用
            graph = self._apply_preprocessing(graph)
            graph = self._apply_k_hop(graph, traj_index)
            graph = self._may_remove_edges_attr(graph)
            graph = self._add_random_edges(graph)
            selected_indices = self._get_masked_indexes(graph)

            # ensure long dtype for edge_index if present
            if getattr(graph, "edge_index", None) is not None:
                graph.edge_index = graph.edge_index.long()

            self._cache_graph(cache_key, graph, selected_indices)

        # 呼び出し側が安全に変更できるようにクローンして返す
        graph_out = graph.clone()
        selected_out = selected_indices.clone() if selected_indices is not None else None
        return graph_out, selected_out

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """
        Global index を受け取り、対応するフレームの graph（と optional な selected_indices）を返す。
        use_previous_data=True の場合は previous_data を graph に付与する。
        """
        traj_index, frame = self.get_traj_frame(index=index)
        traj_number = self.datasets_index[traj_index]
        traj = self._get_trajectory(traj_number)

        # If previous-data is requested, we must ensure that the
        # preprocessing pipeline (which may depend on graph.previous_data)
        # runs with previous data available. The original implementation
        # applied preprocessing first and attached previous_data afterwards,
        # which prevented extra_node_features (e.g. build_features) from
        # seeing the previous frame. To fix this, when use_previous_data=True
        # we build a raw previous-data dict, attach it to the raw graph, and
        # then run the preprocessing pipeline so that the resulting
        # `graph.x` and `graph.previous_data` have matching column layout.

        if not (self.use_previous_data and frame > 0):
            # fast path: no previous-data needed, reuse existing helper
            graph, selected_indices = self._get_processed_graph(traj_index=traj_index, frame=frame, traj=traj)
        else:
            # Build raw graph (unprocessed) for current frame
            raw_graph = get_frame_as_graph(
                traj=traj,
                frame=frame,
                meta=self.meta,
                targets=self.targets,
                frame_target=frame + 1,
            )

            # Construct a previous-data mapping using the same keys the
            # raw trajectory provides. This is a lightweight dict of numpy
            # arrays that extra_node_features (like aneurysm.build_features)
            # can inspect. We include all point-wise arrays except mesh/cells
            prev_dict: Dict[str, np.ndarray] = {}
            try:
                for k in traj.keys():
                    if k in ("mesh_pos", "cells"):
                        continue
                    # select previous frame slice if available
                    try:
                        val = traj[k][frame - 1]
                        prev_dict[k] = val
                    except Exception:
                        # skip keys we cannot slice
                        continue
            except Exception:
                prev_dict = {}

            # Attach previous_data (dict of numpy arrays) so preprocessing
            # functions can consume it when building features.
            raw_graph.previous_data = prev_dict

            # Now apply the standard processing pipeline (preprocessing,
            # k-hop, edge features, random edges, masking). This will
            # consult raw_graph.previous_data where appropriate.
            graph = self._apply_preprocessing(raw_graph)
            graph = self._apply_k_hop(graph, traj_index)
            graph = self._may_remove_edges_attr(graph)
            graph = self._add_random_edges(graph)
            selected_indices = self._get_masked_indexes(graph)

            # ensure long dtype for edge_index if present
            if getattr(graph, "edge_index", None) is not None:
                graph.edge_index = graph.edge_index.long()

            # Cache the processed graph for future calls
            cache_key = (traj_number, frame)
            self._cache_graph(cache_key, graph, selected_indices)

        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        return graph

    def __del__(self):
        """オブジェクト破棄時にファイルハンドルを閉じる（安全に実行）。"""
        try:
            self._close_file_handles()
        except Exception:
            pass