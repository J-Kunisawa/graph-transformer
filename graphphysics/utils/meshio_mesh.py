import os
import shutil
from typing import List, Tuple

import meshio
import numpy as np
from torch_geometric.data import Data


def _ensure_vertices_3d(vertices: np.ndarray) -> np.ndarray:
    if vertices.ndim != 2 or vertices.shape[1] < 1:
        raise ValueError("vertices must be shape (N, D)")
    if vertices.shape[1] < 3:
        padding = np.zeros((vertices.shape[0], 3 - vertices.shape[1]), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding])
    elif vertices.shape[1] > 3:
        raise ValueError(f"Unsupported vertex dimension: {vertices.shape[1]}")
    return vertices


def _extract_faces_from_graph(graph: Data) -> Tuple[str, np.ndarray]:
    """
    Return (cell_type, faces_array) where faces_array is shape (M, K) suitable for meshio.
    Supports graph.face and graph.tetra with common PyG shapes.
    """
    if getattr(graph, "tetra", None) is not None:
        arr = graph.tetra.cpu().numpy()
        # common shapes: (4, M) or (M,4)
        if arr.ndim == 2 and arr.shape[0] in (3, 4) and arr.shape[1] != arr.shape[0]:
            arr = arr.T
        return "tetra", arr
    if getattr(graph, "face", None) is not None:
        arr = graph.face.cpu().numpy()
        # common shapes: (3, M) or (M,3)
        if arr.ndim == 2 and arr.shape[0] in (3, 4) and arr.shape[1] != arr.shape[0]:
            arr = arr.T
        return "triangle", arr
    raise ValueError("Graph must have 'face' or 'tetra' attribute")


def convert_to_meshio_vtu(graph: Data, add_all_data: bool = False) -> meshio.Mesh:
    """
    Converts a PyTorch Geometric Data graph to a meshio.Mesh (VTU-compatible).
    Raises clear errors for missing required attributes.
    """
    if not hasattr(graph, "pos") or graph.pos is None:
        raise ValueError("Graph must have 'pos' attribute with node positions.")

    vertices = graph.pos.cpu().numpy()
    vertices = _ensure_vertices_3d(vertices)

    cell_type, faces = _extract_faces_from_graph(graph)

    # meshio expects cells as list of tuples (name, ndarray)
    cells = [(cell_type, faces)]

    mesh = meshio.Mesh(points=vertices, cells=cells)

    if add_all_data:
        if hasattr(graph, "x") and graph.x is not None:
            x_data = graph.x.cpu().numpy()
            for i in range(x_data.shape[1]):
                mesh.point_data[f"x{i}"] = x_data[:, i]
        if hasattr(graph, "y") and graph.y is not None:
            y_data = graph.y.cpu().numpy()
            for i in range(y_data.shape[1]):
                mesh.point_data[f"y{i}"] = y_data[:, i]

    return mesh


def vtu_to_xdmf(
    filename: str, files_list: List[str], timestep=1, remove_vtus: bool = True
) -> None:
    """
    Writes a time series of VTU meshes into XDMF/HDF5 format.
    - filename: base path without extension
    - files_list: list of VTU file paths (assumed same topology)
    """
    if not files_list:
        raise ValueError("files_list must contain at least one file")

    xdmf_filename = f"{filename}.xdmf"
    h5_filename = f"{filename}.h5"

    # read initial mesh to get points/cells
    init_vtu = meshio.read(files_list[0])
    points = init_vtu.points
    cells = init_vtu.cells

    try:
        with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
            writer.write_points_cells(points, cells)
            t = 0
            for file in files_list:
                mesh = meshio.read(file)
                point_data = getattr(mesh, "point_data", {}) or {}
                cell_data = getattr(mesh, "cell_data", {}) or {}
                writer.write_data(t, point_data=point_data, cell_data=cell_data)
                t += timestep
    except Exception as e:
        # cleanup partial files if any
        if os.path.exists(xdmf_filename):
            try:
                os.remove(xdmf_filename)
            except Exception:
                pass
        raise RuntimeError(f"Failed to write XDMF time series: {e}")

    # meshio writes the .h5 companion in cwd; move it next to xdmf if present
    try:
        generated_h5 = os.path.basename(h5_filename)
        src_h5 = os.path.join(os.getcwd(), generated_h5)
        if os.path.exists(src_h5):
            shutil.move(src_h5, h5_filename)
    except Exception:
        # best-effort move; ignore failures
        pass

    if remove_vtus:
        for file in files_list:
            try:
                os.remove(file)
            except Exception:
                pass