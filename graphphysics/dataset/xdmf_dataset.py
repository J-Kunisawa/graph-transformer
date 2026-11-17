import os
from typing import Callable, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.torch_graph import meshdata_to_graph


class XDMFDataset(BaseDataset):
    """
    XDMF形式（XMLベースのメタデータとHDF5などのバイナリを組み合わせた時系列メッシュフォーマット）
    のシミュレーションデータを読み込み、PyTorch Geometric用のグラフデータとして提供するクラス。
    """
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
    ):
        # 親クラス(BaseDataset)の初期化を呼び出し、インデックス管理や共通パラメータを設定
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            use_partitioning=use_partitioning,
            num_partitions=num_partitions,
            max_nodes_per_partition=max_nodes_per_partition,
        )

        # Keep BaseDataset's CPU device: loader workers must not create CUDA
        # contexts or hold prefetched graphs in GPU memory. Lightning transfers
        # the collated batch to the training device in the main process.
        self.type = "xdmf"

        # シミュレーションの時間刻み幅（dt）を取得
        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "データセットのタイムステップ(dt)が0に設定されています。xdmf形式で保存できるよう dt=1 にフォールバックします。"
            )

        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path

        # 指定フォルダ直下の .xdmf を名前順で読み、軌跡順を再現可能にする
        if not os.path.isdir(xdmf_folder):
            raise FileNotFoundError(f"XDMFフォルダが見つかりません: {xdmf_folder}")
        self.file_paths: List[str] = sorted(
            os.path.join(xdmf_folder, filename)
            for filename in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, filename))
            and filename.endswith(".xdmf")
        )
        if not self.file_paths:
            raise FileNotFoundError(f"XDMFファイルが見つかりません: {xdmf_folder}")
        self._size_dataset = len(self.file_paths)

        # 見つけたすべてのXDMFファイルについて、ノード数を読み取ってインデックスマップを構築する
        self._build_index_map()

    def _build_index_map(self):
        """
        各XDMFファイル（軌跡）を開き、メッシュのノード数を取得して、
        BaseDatasetのグローバルインデックス（何番目のデータがどのファイルの何フレーム目か）を構築する。
        """
        for traj_index, file_path in enumerate(self.file_paths):
            # meshioを使ってXDMFファイルの構造を読み取る
            with meshio.xdmf.TimeSeriesReader(file_path) as reader:
                points, _ = reader.read_points_cells()
                num_nodes = len(points)
            self._add_traj_to_index_map(traj_index, num_nodes)

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """
        DataLoaderから呼ばれるメインメソッド。
        指定されたグローバルインデックスに対応するフレームのグラフデータを読み込み、前処理して返す。

        Parameters:
            index (int): データセット内のグローバルインデックス。

        Returns:
            Union[Data, Tuple[Data, torch.Tensor]]: 指定されたフレームのグラフデータ。
            マスキングが適用されている場合は、マスク対象のインデックスも併せて返す。
        """
        # グローバルインデックスから、軌跡(ファイル番号)、フレーム(時刻)、サブグラフ番号を逆算
        traj_index, frame, subgraph_idx = self._get_indices(index)
        xdmf_file = self.file_paths[traj_index]
        
        # ファイル名からメッシュのID（識別子）を抽出 (例: "result_MESH_11.xdmf" -> "11")
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]

        # 対象のXDMFファイルから指定フレームのデータを読み込む
        with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
            num_steps = reader.num_steps
            # ターゲット(1ステップ先のデータ)を取得するため、最後のフレームは入力として使えない
            if frame >= num_steps - 1:
                raise IndexError(
                    f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
                )

            # メッシュの座標(points)と接続情報(cells)を読み込み
            points, cells = reader.read_points_cells()
            
            # 現在のフレーム（t）の物理量を読み込み
            time, point_data, _ = reader.read_data(frame)
            # 教師データとなる次のフレーム（t+1）の物理量を読み込み
            _, target_point_data, _ = reader.read_data(frame + 1)

            # 加速度などを計算するために過去のフレーム（t-1）が必要な場合の読み込み
            if self.use_previous_data:
                _, previous_data, _ = reader.read_data(frame - 1)

        # meshioのMeshオブジェクトとして一時的に構築
        mesh = meshio.Mesh(points, cells, point_data=point_data)

        # グラフのエッジを作るためにセル(要素)のタイプを確認し抽出
        # 現在は2Dの三角形(triangle)か、3Dの四面体(tetra)のみをサポート
        if "triangle" in mesh.cells_dict:
            cells = mesh.cells_dict["triangle"]
        elif "tetra" in mesh.cells_dict:
            cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
        else:
            raise ValueError(
                "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
            )

        # メタデータ(JSON)で定義されている必要な物理量のみを抽出し、型(dtype)を合わせる
        point_data = {
            k: np.array(mesh.point_data[k]).astype(self.meta["features"][k]["dtype"])
            for k in self.meta["features"]
            if k in mesh.point_data.keys()
        }

        # ターゲットデータ（予測対象）と、次ステップの入力用データの仕分け
        target_data = {}
        next_data = {}
        for k in self.meta["features"]:
            if k in self.targets:
                # モデルが予測すべき対象（例: velocity）
                target_data[k] = np.array(target_point_data[k]).astype(
                    self.meta["features"][k]["dtype"]
                )
            else:
                # 予測対象ではないが、時間変化するフィールド（境界条件など）は、
                # 次のステップの自律回帰予測の際に入力として利用できるように next_data に格納しておく
                if (
                    k in target_point_data.keys()
                    and self.meta["features"][k]["type"] == "dynamic"
                ):
                    next_data[k] = np.array(target_point_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )

        # 1次元配列(N,)を2次元の列ベクトル(N, 1)に変換する内部ヘルパー関数
        def _reshape_array(a: dict):
            for k, v in a.items():
                if v.ndim == 1:
                    a[k] = v.reshape(-1, 1)

        _reshape_array(point_data)
        _reshape_array(target_data)

        # 生のメッシュデータ(numpy)から、PyTorch GeometricのDataオブジェクト(グラフ)を生成
        graph = meshdata_to_graph(
            points=points.astype(np.float32),
            cells=cells,
            point_data=point_data,
            time=time,
            target=target_data,
            id=mesh_id,
            next_data=next_data,
        )

        # 過去データが存在する場合も整形してグラフの属性として持たせる
        if self.use_previous_data:
            previous_dict = {
                k: np.array(previous_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in previous_data.keys()
                and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_array(previous_dict)

            # build_features がテンソルを期待するため、H5Datasetと同様にテンソルへ変換
            # 物理量辞書を順番に結合し、最後に時刻（t-1）のカラムを追加
            prev_arrays = [arr.astype(np.float32) for arr in previous_dict.values()]
            prev_tensor = torch.from_numpy(np.concatenate(prev_arrays, axis=1))

            prev_time = (frame - 1) * self.dt
            time_col = torch.full((prev_tensor.size(0), 1), prev_time, dtype=torch.float32)
            # Dataオブジェクトの属性としてTensorをアタッチ（デバイス転送前に行う）
            graph["previous_data"] = torch.cat([prev_tensor, time_col], dim=1)

        # デバイス（GPUなど）への転送
        graph = graph.to(self.device)

        # --- ここから BaseDataset に定義された前処理パイプラインの適用 ---
        graph = self._apply_preprocessing(graph) # カスタム特徴量の計算やガウスノイズ付加
        graph = self._apply_k_hop(graph, traj_index) # K-hopエッジの拡張
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph) # ランダムジャンプエッジの追加
        selected_indices = self._get_masked_indexes(graph) # ノードのマスキング

        # エッジインデックスは必ずLong型(整数)にする
        graph.edge_index = (
            graph.edge_index.long() if graph.edge_index is not None else None
        )

        # 前処理（_apply_preprocessing）の中で、速度差分(加速度)等の計算に past_data や next_data が
        # 利用された後、メモリ節約と DataLoader のバッチ化エラーを防ぐためにこれらを削除する。
        if hasattr(graph, "next_data"):
            del graph.next_data
        if "previous_data" in graph:
            del graph["previous_data"]
        
        graph.traj_index = traj_index

        # TODO: マスキング(selected_indices)とメッシュ分割(Partitioning)の併用は未対応
        # 指定されたサブグラフ領域（分割されたメッシュの一部）のみを切り出す
        if self.use_partitioning:
            graph = self._get_partition(graph, traj_index, subgraph_idx)

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph