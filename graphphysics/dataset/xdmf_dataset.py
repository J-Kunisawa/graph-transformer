import os
import random
from pathlib import Path
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
    XDMF (time series) ベースのデータセット。

    - xdmf_folder: XDMF ファイルが置かれたディレクトリ（各ファイルが1軌跡）
    - targets: 予測対象フィールド名のリスト（BaseDataset の targets に渡す）
    - random_prev/random_next: previous/target のフレームオフセットをランダムに選ぶ幅
    - switch_to_val: True の場合 train -> test 置換（ファイル名/フォルダ規則に依存）
    """

    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: Optional[List[str]] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        random_prev: int = 1,
        random_next: int = 1,
    ) -> None:
        super().__init__(
            meta_path=meta_path,
            targets=targets or [],
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = "xdmf"

        self.random_prev = int(random_prev)
        self.random_next = int(random_next)

        # フォルダ切替（train <-> test）
        folder = xdmf_folder
        if switch_to_val:
            folder = folder.replace("train", "test")
            self.random_prev = 1
            self.random_next = 1

        self.xdmf_folder = folder
        self.meta_path = meta_path

        p = Path(self.xdmf_folder)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"XDMF folder not found: {self.xdmf_folder}")

        # XDMF ファイル一覧（安定ソート）
        self.file_paths: List[str] = sorted(
            [str(p / f) for f in os.listdir(self.xdmf_folder) if f.endswith(".xdmf")]
        )
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .xdmf files found in {self.xdmf_folder}")

        self._size_dataset: int = len(self.file_paths)

        # dt のフォールバックは BaseDataset の meta を使う
        self.dt = float(self.meta.get("dt", 1.0))
        if self.dt == 0:
            self.dt = 1.0
            logger.warning("meta.dt == 0, fallback to dt=1.0")

    @property
    def size_dataset(self) -> int:
        return int(self._size_dataset)

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """
        Global index から (trajectory_index, frame) を決定し、
        指定フレームの graph を返す。masking が有効なら (graph, selected_indices) を返す。
        """
        traj_index, frame = self.get_traj_frame(index=index)
        xdmf_file = self.file_paths[traj_index]

        # ランダムに previous / target オフセットを決定（範囲内に収める）
        target_offset = random.randint(1, max(1, self.random_next))
        previous_offset = random.randint(1, max(1, self.random_prev))

        # Read XDMF time series
        with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
            num_steps = reader.num_steps
            # frame 範囲チェック
            if frame < 0 or frame >= num_steps - 1:
                raise IndexError(
                    f"Frame {frame} out of bounds for {xdmf_file} (num_steps={num_steps})"
                )

            # clamp offsets so that indices remain valid
            if frame - previous_offset < 0:
                previous_offset = min(previous_offset, frame)
            if frame + target_offset > num_steps - 1:
                target_offset = 1

            # read base frame and target frame data
            points, cells = reader.read_points_cells()
            time, point_data, _ = reader.read_data(frame)
            _, target_point_data, _ = reader.read_data(frame + target_offset)

            previous_point_data = None
            if self.use_previous_data and (frame - previous_offset) >= 0:
                _, previous_point_data, _ = reader.read_data(frame - previous_offset)

        # Prepare mesh and cells: support triangle/tetra
        mesh = meshio.Mesh(points, cells, point_data=point_data)
        if "triangle" in mesh.cells_dict:
            cells_arr = mesh.cells_dict["triangle"]
        elif "tetra" in mesh.cells_dict:
            cells_arr = np.array(mesh.cells_dict["tetra"], dtype=np.int64)
        else:
            raise ValueError("Unsupported cell type in XDMF: need 'triangle' or 'tetra'")

        # Collect point_data arrays according to meta features
        point_data_dict = {
            k: np.asarray(mesh.point_data[k]).astype(self.meta["features"][k]["dtype"])
            for k in self.meta["features"]
            if k in mesh.point_data
        }

        # Build target and next_data dicts
        target_data: dict = {}
        next_data: dict = {}
        for k in self.meta["features"]:
            if k in target_point_data:
                if k in self.targets:
                    target_data[k] = np.asarray(target_point_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )
                else:
                    # dynamic なフィールドとして next_data に入れる
                    if self.meta["features"][k]["type"] == "dynamic":
                        next_data[k] = np.asarray(target_point_data[k]).astype(
                            self.meta["features"][k]["dtype"]
                        )

        # reshape 1D -> (N,1)
        def _reshape_arrays(d: dict) -> None:
            for kk, vv in list(d.items()):
                arr = np.asarray(vv)
                if arr.ndim == 1:
                    d[kk] = arr.reshape(-1, 1)
                else:
                    d[kk] = arr

        _reshape_arrays(point_data_dict)
        _reshape_arrays(target_data)
        _reshape_arrays(next_data)

        # Convert mesh+point_data -> torch_geometric.data.Data
        graph = meshdata_to_graph(
            points=points.astype(np.float32),
            cells=cells_arr,
            point_data=point_data_dict,
            time=time,
            target=target_data,
            id=os.path.splitext(os.path.basename(xdmf_file))[0],
            next_data=next_data,
        )

        # attach dt info for target/previous
        graph.target_dt = target_offset * self.dt

        if self.use_previous_data and previous_point_data is not None:
            prev_dict = {
                k: np.asarray(previous_point_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in previous_point_data and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_arrays(prev_dict)
            graph.previous_data = prev_dict
            graph.previous_dt = -previous_offset * self.dt

        # Move to configured device and apply dataset transforms
        graph = graph.to(self.device)
        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph)
        selected_indices = self._get_masked_indexes(graph)

        # ensure edge_index dtype
        if getattr(graph, "edge_index", None) is not None:
            graph.edge_index = graph.edge_index.long()

        # cleanup temporary fields not needed downstream
        if hasattr(graph, "next_data"):
            del graph.next_data

        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        return graph