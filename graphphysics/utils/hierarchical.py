import json
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from torch_geometric.data import Data

from graphphysics.utils.torch_graph import meshdata_to_graph


def read_h5_metadata(
    dataset_path: str, meta_path: str
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Reads trajectory indices and metadata without keeping the file open."""
    with h5py.File(dataset_path, "r") as file_handle:
        datasets_index = list(file_handle.keys())

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    return datasets_index, len(datasets_index), meta


def get_h5_dataset(
    dataset_path: str, meta_path: str
) -> Tuple[h5py.File, List[str], int, Dict[str, Any]]:
    """
    Open an HDF5 file and return (file_handle, datasets_index, n_datasets, meta).
    Caller is responsible for closing the returned file handle.
    """
    file_handle = h5py.File(dataset_path, "r")
    datasets_index = list(file_handle.keys())
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return file_handle, datasets_index, len(datasets_index), meta


def get_traj_as_meshes(
    file_handle: h5py.File, traj_number: str, meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Load all feature arrays for a given trajectory into a dict of numpy arrays.

    Each feature is converted to the dtype specified in meta["features"][name]["dtype"]
    and reshaped according to meta["features"][name]["shape"].
    Missing features raise a KeyError.
    """
    if traj_number not in file_handle:
        raise KeyError(f"Trajectory '{traj_number}' not found in HDF5 file")

    features_group = file_handle[traj_number]
    meshes: Dict[str, np.ndarray] = {}

    for key, spec in meta.get("features", {}).items():
        if key not in features_group:
            raise KeyError(f"Feature '{key}' not found in trajectory '{traj_number}'")
        arr = features_group[key][()]
        arr = np.asarray(arr).astype(spec["dtype"], copy=False)
        # reshape if a shape is provided
        shape = spec.get("shape", None)
        if shape is not None:
            arr = arr.reshape(shape)
        meshes[key] = arr

    return meshes


def get_frame_as_mesh(
    traj: Dict[str, np.ndarray],
    frame: int,
    targets: Optional[List[str]] = None,
    frame_target: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
]:
    """
    Extract mesh_pos, cells, point_data (for `frame`), optional target_point_data (for frame_target),
    and optional next_data (other dynamic fields from frame_target).

    Returns a 5-tuple:
      (mesh_pos, cells, point_data, target_point_data, next_data)
    target_point_data and next_data are None when frame_target is None.
    """
    if targets is None:
        targets = []

    # mesh positions and cells: handle (T, N, D) or (N, D) shapes
    mesh_pos_arr = traj.get("mesh_pos")
    if mesh_pos_arr is None:
        raise KeyError("mesh_pos missing in trajectory data")
    mesh_pos = mesh_pos_arr[frame] if getattr(mesh_pos_arr, "ndim", 0) > 1 else mesh_pos_arr

    cells_arr = traj.get("cells")
    if cells_arr is None:
        raise KeyError("cells missing in trajectory data")
    cells = cells_arr[frame] if getattr(cells_arr, "ndim", 0) > 1 else cells_arr

    # collect point-wise features for the frame (exclude mesh_pos, cells, node_type)
    point_data: Dict[str, np.ndarray] = {}
    for key, val in traj.items():
        if key in ("mesh_pos", "cells", "node_type"):
            continue
        # if per-frame array, index it; else take it as-is
        point_data[key] = val[frame] if getattr(val, "ndim", 0) > 1 else val

    # node_type: often stored as shape (1, N) or (N,)
    node_type_arr = traj.get("node_type")
    if node_type_arr is None:
        raise KeyError("node_type missing in trajectory data")
    node_type = node_type_arr[0] if getattr(node_type_arr, "ndim", 0) > 1 else node_type_arr
    point_data["node_type"] = node_type

    target_point_data: Optional[Dict[str, np.ndarray]] = None
    next_data: Optional[Dict[str, np.ndarray]] = None
    if frame_target is not None:
        target_point_data = {}
        next_data = {}
        for key, val in traj.items():
            if key in ("mesh_pos", "cells", "node_type"):
                continue
            if key in targets:
                target_point_data[key] = val[frame_target] if getattr(val, "ndim", 0) > 1 else val
            else:
                next_data[key] = val[frame_target] if getattr(val, "ndim", 0) > 1 else val

    return mesh_pos, cells, point_data, target_point_data, next_data


def get_frame_as_graph(
    traj: Dict[str, np.ndarray],
    frame: int,
    meta: Dict[str, Any],
    targets: Optional[List[str]] = None,
    frame_target: Optional[int] = None,
) -> Data:
    """
    Convert a single frame (and optional target frame) from trajectory dict into a PyG Data graph.
    """
    points, cells, point_data, target, next_data = get_frame_as_mesh(traj, frame, targets, frame_target)
    time = float(frame) * float(meta.get("dt", 1))
    return meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        time=time,
        target=target,
        next_data=next_data,
        meta=meta,
    )