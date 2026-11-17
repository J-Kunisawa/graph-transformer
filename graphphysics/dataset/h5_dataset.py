from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

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

# HDF5形式のファイルを読み込むためのライブラリ。インストールされていない場合はエラーを出す
try:
    import h5py
except ImportError as exc:
    raise RuntimeError("H5Datasetを使用するにはh5pyが必要です") from exc


class H5Dataset(BaseDataset):
    """
    HDF5 (.h5) 形式の物理シミュレーションデータを読み込み、PyTorch Geometricのグラフデータ(Data)
    として提供するデータセットクラス。BaseDatasetを継承し、インデックス管理や前処理を引き継ぐ。
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
        world_pos_parameters: Optional[dict] = None,
        cache_size: int = 8,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
    ):
        # 親クラス(BaseDataset)の初期化を呼び出し、共通設定やインデックス管理機能をセットアップ
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
            use_partitioning=use_partitioning,
            num_partitions=num_partitions,
            max_nodes_per_partition=max_nodes_per_partition,
        )

        self.type = "h5"
        self.h5_path = h5_path
        self.meta_path = meta_path
        
        # メモリ（RAM）に保持する軌跡やグラフの最大数（ディスクI/Oを減らし高速化するため）
        self.cache_size = cache_size

        # シミュレーションの時間刻み幅（dt）を取得
        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "データセットのタイムステップ(dt)が0に設定されています。xdmf形式で保存できるよう dt=1 にフォールバックします。"
            )

        # HDF5ファイルとメタデータから、データセット内の全軌跡（シミュレーションケース）の情報を取得
        (
            self.datasets_index, # 軌跡の名前（キー）のリスト
            self._size_dataset,  # 軌跡の総数
            self.meta,           # 補完されたメタデータ
        ) = read_h5_metadata(dataset_path=h5_path, meta_path=meta_path)

        # PyTorchのDataLoaderが複数ワーカー(num_workers > 0)で動く際、
        # 各プロセスが独自のHDF5ファイルハンドルを持てるようにするための辞書
        self._file_handles: dict[str, h5py.File] = {}
        
        # LRU（Least Recently Used）キャッシュとして機能するOrderedDict。
        # 古いデータから順に破棄してメモリ枯渇を防ぐ。
        self._trajectory_cache: OrderedDict[str, dict] = OrderedDict()
        # 確率的な前処理結果ではなく、生グラフだけをキャッシュする。
        self._frame_cache: OrderedDict[tuple[str, int], Data] = OrderedDict()
        
        # BaseDatasetのインデックスマップ構築処理を実行
        self._build_index_map()

    def _build_index_map(self):
            """
            HDF5内の各軌跡（シミュレーション動画）のノード数を取得し、
            親クラスの `_add_traj_to_index_map` を使ってグローバルインデックスを構築する。
            """
            handle = self._get_file_handle()
            for traj_index, traj_name in enumerate(self.datasets_index):
                # メッシュのノード数を取得
                num_nodes = handle[traj_name]["mesh_pos"].shape[-2]
                self._add_traj_to_index_map(traj_index, num_nodes)

            # 初期化が終わったこの段階で、一旦すべてのファイルハンドルを綺麗に閉じてリセットする
            self._close_file_handles()
            
    def _close_file_handles(self):
        """開いている全てのHDF5ファイルハンドルを安全に閉じる"""
        for handle in self._file_handles.values():
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        self._file_handles.clear()

    def _get_file_handle(self) -> h5py.File:
        """
        現在のプロセス（ワーカー）に対応するHDF5ファイルハンドルを取得する。
        ※マルチプロセス環境でHDF5を安全に読み込むための必須処理。
        """
        worker_info = get_worker_info()
        worker_id = str(worker_info.id) if worker_info is not None else "main"
        handle = self._file_handles.get(worker_id)
        
        # まだこのワーカー用のハンドルが開かれていなければ、Read-Only("r")で開く
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._file_handles[worker_id] = handle
        return handle

    def _get_trajectory(self, traj_number: str) -> dict:
        """
        指定された軌跡（シミュレーションケース全体）のデータをキャッシュから取得するか、
        HDF5から読み込んでキャッシュに保存する。
        """
        cached = self._trajectory_cache.get(traj_number)
        if cached is not None:
            # キャッシュヒットした場合は、最新の使用履歴として末尾に移動（LRU管理）
            self._trajectory_cache.move_to_end(traj_number)
            return cached

        # キャッシュになければHDF5から読み込む
        file_handle = self._get_file_handle()
        trajectory = get_traj_as_meshes(
            file_handle=file_handle, traj_number=traj_number, meta=self.meta
        )
        
        # 読み込んだデータをキャッシュに保存し、上限を超えたら一番古いものを削除
        self._trajectory_cache[traj_number] = trajectory
        if len(self._trajectory_cache) > self.cache_size:
            self._trajectory_cache.popitem(last=False)
        return trajectory

    def _cache_graph(
        self,
        key: tuple[str, int],
        graph: Data,
    ) -> None:
        """確定的な前処理前のグラフをメモリにキャッシュする。"""
        self._frame_cache[key] = graph
        self._frame_cache.move_to_end(key)
        
        # グラフキャッシュは軌跡キャッシュの2倍の容量を持たせる設定
        if len(self._frame_cache) > self.cache_size * 2:
            self._frame_cache.popitem(last=False)
        return None

    def _build_node_features(self, traj: dict, frame: int) -> torch.Tensor:
        """
        指定されたフレームの物理量（速度や圧力など）を辞書から取り出し、
        1つの特徴量テンソル（Node Features Matrix）として結合する。
        """
        # シミュレーション上の現在時刻を計算
        time = frame * self.meta.get("dt", 1)

        # 座標やセル構成、ノードタイプ以外の、動的な物理量（ポイントデータ）を抽出
        point_data = {
            key: (traj[key][frame] if traj[key].ndim > 1 else traj[key])
            for key in traj.keys()
            if key not in ["mesh_pos", "cells", "node_type"]
        }
        # ノードタイプ（境界条件など）は時間変化しないため最初のフレームのものを取得
        point_data["node_type"] = traj["node_type"][0]

        arrays = []
        # 各フィールドをFloat32のNumPy配列に変換し、リストにまとめる
        for data in point_data.values():
            arr = data
            if arr.ndim == 1:
                arr = arr[:, None] # 1次元配列なら2次元に拡張 (N,) -> (N, 1)
            arrays.append(arr.astype(np.float32))

        # 特徴量を横方向（チャンネル方向）に結合
        if arrays:
            node_features = np.concatenate(arrays, axis=1)
        else:
            node_features = np.zeros((traj["mesh_pos"].shape[-2], 0), dtype=np.float32)

        # 特徴量の最後に「時刻(time)」の列を追加する
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
        指定フレームのグラフを取得し、ノイズ付加やK-hopなどの前処理を適用する。
        生グラフだけをキャッシュし、確率的な前処理はアクセスごとに再実行する。
        """
        traj_number = self.datasets_index[traj_index]
        cache_key = (traj_number, frame)

        # キャッシュの確認
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            base_graph = cached
        else:
            # キャッシュがない場合、ゼロからグラフを構築する
            if traj is None:
                traj = self._get_trajectory(traj_number)

            # 辞書データからPyTorch GeometricのDataオブジェクト(グラフ)を作成
            # (ここで x, y, pos, edge_index が構築される)
            base_graph = get_frame_as_graph(
                traj=traj,
                frame=frame,
                meta=self.meta,
                targets=self.targets,
                frame_target=frame + 1, # 次のフレームの変化量がターゲット(教師ラベル)
            )

            # 前処理（build_features等）で過去データが必要な場合に備え、内部ストレージにセット
            if self.use_previous_data:
                previous_features = self._build_node_features(traj, frame - 1)
                base_graph.previous_data = previous_features

            self._cache_graph(cache_key, base_graph)

        # キャッシュ本体を確率的な前処理で破壊しないよう、先にクローンする。
        graph = base_graph.clone()
        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph)
        selected_indices = self._get_masked_indexes(graph)

        graph.edge_index = (
            graph.edge_index.long() if graph.edge_index is not None else None
        )
        return graph, selected_indices

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """
        DataLoaderから呼ばれるメインのデータ取得メソッド。
        1次元のインデックスから対応する軌跡とフレームを逆算し、グラフを返す。

        パラメータ:
            index (int): データセット内のグローバルインデックス

        戻り値:
            Union[Data, Tuple[Data, torch.Tensor]]: 指定されたフレームのグラフデータ。
            マスキングが適用されている場合は、選択されたインデックスも一緒に返す。
        """
        # 親クラスのメソッドでインデックスを分解
        traj_index, frame, subgraph_idx = self._get_indices(index)
        traj_number = self.datasets_index[traj_index]
        
        # 軌跡データ（辞書）をロードまたはキャッシュから取得
        traj = self._get_trajectory(traj_number)

        # 前処理済みのグラフデータを取得
        graph, selected_indices = self._get_processed_graph(
            traj_index=traj_index, frame=frame, traj=traj
        )

        graph.traj_index = traj_index

        # TODO: マスキング(selected_indices)とパーティショニングの併用は現在未対応
        # メッシュ分割（メモリ節約用）が有効な場合、指定されたサブ領域のグラフを切り出す
        if self.use_partitioning:
            graph = self._get_partition(graph, traj_index, subgraph_idx)

        # 戻り値のフォーマット
        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph

    def __del__(self):
        """
        オブジェクトが破棄される際、HDF5ファイルが開きっぱなしにならないよう確実に閉じる。
        """
        self._close_file_handles()