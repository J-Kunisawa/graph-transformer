import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch_geometric.transforms as T
from loguru import logger
from meshio import Mesh
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import negative_sampling

# サポートするメッシュ分割アルゴリズムの定義（現在はMETISのみ）
PARTITION_METHODS = ["metis"]


def merge_edge_indices(
    edge_indices: Sequence[Optional[torch.Tensor]], num_nodes: int
) -> torch.Tensor:
    """複数の edge_index を結合し、重複を除去して返す。"""
    valid = [edges for edges in edge_indices if edges is not None and edges.numel() > 0]
    if not valid:
        return torch.empty((2, 0), dtype=torch.long)

    device = valid[0].device
    merged = torch.cat([edges.to(device=device, dtype=torch.long) for edges in valid], dim=1)
    values = torch.ones(merged.size(1), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        merged, values, size=(num_nodes, num_nodes), device=device
    ).coalesce().indices()


def _batch_ranges(
    batch: Optional[torch.Tensor], num_nodes: int, device: torch.device
) -> List[tuple[int, int]]:
    """PyG Batch の連続したノード範囲を ``(start, end)`` で返す。"""
    if batch is None:
        return [(0, num_nodes)]
    if batch.numel() != num_nodes:
        raise ValueError("batch の要素数は num_nodes と一致する必要があります。")

    batch = batch.to(device=device, dtype=torch.long)
    if batch.numel() > 1 and torch.any(batch[1:] < batch[:-1]):
        raise ValueError("batch はPyG Batchと同様にグラフごとに連続している必要があります。")

    _, counts = torch.unique_consecutive(batch, return_counts=True)
    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(dim=0),
        ]
    )
    return [
        (int(boundaries[i].item()), int(boundaries[i + 1].item()))
        for i in range(counts.numel())
    ]


def compute_exact_k_hop_edge_index(
    edge_index: torch.Tensor,
    num_hops: int,
    num_nodes: int,
) -> torch.Tensor:
    """ちょうど ``num_hops`` 歩で到達可能な疎な隣接関係 ``A^k`` を返す。

    ``compute_k_hop_edge_index`` が ``A + ... + A^k`` を返すのに対し、
    本関数はDilated Attention用に ``A^k`` の構造だけを返す。
    """
    if num_hops < 1:
        raise ValueError("num_hops は1以上である必要があります。")

    device = edge_index.device
    edge_index = merge_edge_indices([edge_index], num_nodes=num_nodes)
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(
        edge_index, values, size=(num_nodes, num_nodes), device=device
    ).coalesce()
    adjacency_power = adjacency

    for _ in range(num_hops - 1):
        adjacency_power = torch.sparse.mm(adjacency_power, adjacency).coalesce()
        indices = adjacency_power.indices()
        adjacency_power = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.size(1), dtype=torch.float32, device=device),
            size=(num_nodes, num_nodes),
            device=device,
        ).coalesce()

    indices = adjacency_power.indices()
    without_self_loops = indices[0] != indices[1]
    return indices[:, without_self_loops]


def generate_random_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    ratio: float,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """各グラフ内で既存エッジ数に対して ``ratio`` 分のランダム無向エッジを生成する。"""
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("random edge ratio は0以上1以下である必要があります。")
    if ratio == 0.0 or edge_index.numel() == 0:
        return edge_index.new_empty((2, 0), dtype=torch.long)

    device = edge_index.device
    ranges = _batch_ranges(batch, num_nodes=num_nodes, device=device)
    if batch is not None:
        local_batch = batch.to(device=device, dtype=torch.long)
        if torch.any(local_batch[edge_index[0]] != local_batch[edge_index[1]]):
            raise ValueError("異なるバッチ要素を結ぶ既存エッジは使用できません。")

    generated: List[torch.Tensor] = []
    for start, end in ranges:
        node_count = end - start
        if node_count < 2:
            continue
        mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        local_edges = edge_index[:, mask] - start
        requested = int(math.ceil(local_edges.size(1) * ratio))
        if requested == 0:
            continue
        random_edges = negative_sampling(
            local_edges,
            num_nodes=node_count,
            num_neg_samples=requested,
            method="sparse",
            force_undirected=True,
        )
        if random_edges.numel() > 0:
            generated.append(random_edges + start)

    if not generated:
        return edge_index.new_empty((2, 0), dtype=torch.long)
    return merge_edge_indices(generated, num_nodes=num_nodes).to(device)


def generate_global_attention_edges(
    node_type: torch.Tensor,
    num_nodes: int,
    global_node_types: Sequence[int],
    sample_ratio: float,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """指定種別から抽出したGlobal nodeを同一グラフの全ノードへ双方向接続する。"""
    if sample_ratio < 0.0 or sample_ratio > 1.0:
        raise ValueError("global attention sample_ratio は0以上1以下である必要があります。")
    if sample_ratio == 0.0 or not global_node_types:
        return node_type.new_empty((2, 0), dtype=torch.long)
    if node_type.numel() != num_nodes:
        raise ValueError("node_type の要素数は num_nodes と一致する必要があります。")

    device = node_type.device
    node_type = node_type.reshape(-1).to(dtype=torch.long)
    ranges = _batch_ranges(batch, num_nodes=num_nodes, device=device)
    allowed_types = torch.as_tensor(global_node_types, dtype=torch.long, device=device)
    generated: List[torch.Tensor] = []

    for start, end in ranges:
        local_types = node_type[start:end]
        candidates = torch.nonzero(
            torch.isin(local_types, allowed_types), as_tuple=False
        ).reshape(-1) + start
        if candidates.numel() == 0:
            continue

        selected_count = min(
            candidates.numel(), max(1, int(math.ceil(candidates.numel() * sample_ratio)))
        )
        selected = candidates[
            torch.randperm(candidates.numel(), device=device)[:selected_count]
        ]
        all_nodes = torch.arange(start, end, dtype=torch.long, device=device)
        sources = selected.repeat_interleave(all_nodes.numel())
        destinations = all_nodes.repeat(selected.numel())
        forward = torch.stack([sources, destinations], dim=0)
        reverse = torch.stack([destinations, sources], dim=0)
        edges = torch.cat([forward, reverse], dim=1)
        generated.append(edges[:, edges[0] != edges[1]])

    if not generated:
        return node_type.new_empty((2, 0), dtype=torch.long)
    return merge_edge_indices(generated, num_nodes=num_nodes).to(device)


def compute_k_hop_edge_index(
    edge_index: torch.Tensor,
    num_hops: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    疎行列の積（Sparse Matrix Multiplication）を用いて、グラフの k-hop 隣接関係（k歩先のエッジトポロジー）を高速計算する。
    
    隣接行列 A の k 階乗（A^k）を計算することで、k-hop 先に存在するノード同士の繋がりを網羅的に抽出する。
    """
    edge_index_device = edge_index.device
    
    # 1. 元のエッジインデックスから、1ビットの値を保持する「疎な隣接行列 (A)」を組み立てて座標をソート（coalesce）
    adj = torch.sparse_coo_tensor(
        edge_index,
        values=torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index_device
        ),
        size=(num_nodes, num_nodes),
    ).coalesce()

    adj_k = adj.clone()
    
    # 2. 疎行列同士の積を (num_hops - 1) 回反復することで、遠距離の接続行列を計算
    for _ in range(num_hops - 1):
        # A_k = A_k-1 × A
        adj_k = adj_k + torch.sparse.mm(adj_k, adj)
        adj_k = adj_k.coalesce()

        # 【自己ループ（Self-loops）の除去処理】
        # 行列積の過程で、自分自身に戻ってくるエッジ（始点ID == 終点ID）が生成されるため、これらをマスクして削除
        indices = adj_k.indices()
        mask = indices[0] != indices[1] # 始点と終点が異なる場所のみ True
        
        adj_k = torch.sparse_coo_tensor(
            indices=indices[:, mask],
            values=adj_k.values()[mask],
            size=adj_k.size(),
        ).coalesce()

    # 最終的な k-hop 隣接エッジ配列をインデックスとして抽出
    khop_edge_index = adj_k.indices()
    return khop_edge_index


def compute_k_hop_graph(
    graph: Data,
    num_hops: int,
    add_edge_features_to_khop: bool = False,
    device: str = "cpu",
    world_pos_index_start: int = 0,
    world_pos_index_end: int = 3,
) -> Data:
    """
    入力されたグラフを基に、指定されたホップ長（num_hops）に拡張された k-hop メッシュグラフを構築する。
    """
    if num_hops == 1:
        return graph

    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    # 拡張隣接エッジ配列を計算
    khop_edge_index = compute_k_hop_edge_index(
        edge_index=edge_index,
        num_hops=num_hops,
        num_nodes=num_nodes,
    ).to(device)

    # 元のグラフ構造をクローンし、計算したエッジトポロジーで上書き
    khop_mesh_graph = graph.clone()
    khop_mesh_graph.edge_index = khop_edge_index.to(device)

    # オプション：拡張された新しいエッジ群に対し、相対位置ベクトルやユークリッド距離を再計算して割り当て
    if add_edge_features_to_khop:
        transforms = [
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ]
        edge_feature_computer = T.Compose(transforms)
        khop_mesh_graph.edge_attr = None # 一度古い属性をクリア
        khop_mesh_graph = edge_feature_computer(khop_mesh_graph).to(device)

    return khop_mesh_graph


def create_subgraphs(
    graph: Data, num_partitions: int, partition_method: str = "metis"
):
    """
    METIS グラフパーティショニングアルゴリズムを用いて、大規模なメッシュグラフを
    自身のVRAM（GPUメモリ）上限に収まる複数の独立したサブグラフへと分割する。
    """
    if partition_method not in PARTITION_METHODS:
        raise ValueError(f"サポートされていない分割手法です: {partition_method}。利用可能: {PARTITION_METHODS}")
        
    if partition_method == "metis":
        # ClusterData により、エッジの切断数が最小（通信コスト最小）になるようにメッシュを自動分割
        cluster = ClusterData(graph, num_parts=num_partitions, log=False)
        # サブグラフをミニバッチ形式で1つずつロードするための ClusterLoader を立ち上げ
        loader = ClusterLoader(cluster, batch_size=1, shuffle=False)
        
        partition = cluster.partition
        # 分割された各サブグラフ（パーティション）に、元々のグローバルなノードIDが
        # どのように割り振られたかの一覧マップ（node_perm）を作成して取得
        partitioned_node_ids = [
            partition.node_perm[partition.partptr[i] : partition.partptr[i + 1]].cpu()
            for i in range(num_partitions)
        ]
        return loader, partitioned_node_ids


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[dict[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[dict[str, np.ndarray]] = None,
    return_only_node_features: bool = False,
    id: Optional[str] = None,
    next_data: Optional[np.ndarray] = None,
) -> Data:
    """
    NumPy配列の生メッシュ構成データ（座標、セル接続、物理量）をパッキングし、
    PyTorch Geometricの Data グラフオブジェクトを組み上げる統合コア関数。
    """
    # 1. 流れ込んできた全ての点データ（流速、圧力、ノード属性等）を横方向（チャンネル次元）に結合
    if point_data is not None:
        if any(data.ndim > 1 for data in point_data.values()):
            # 特徴量フィールドに多次元配列が含まれる場合（hstackで結合し、最後に現在時刻の列を追加）
            node_features = np.hstack(
                [data for data in point_data.values()]
                + [np.full((len(points),), time).reshape((-1, 1))]
            )
            node_features = torch.tensor(node_features, dtype=torch.float32)
        else:
            # すべてが1次元配列の場合（vstackで縦に重ねてから転置をとる）
            node_features = np.vstack(
                [data for data in point_data.values()] + [np.full((len(points),), time)]
            ).T
            node_features = torch.tensor(node_features, dtype=torch.float32)
    else:
        # 点データが無い場合は、時刻を格納するだけの暫定特徴量マトリクスを作成
        node_features = torch.zeros((len(points), 1), dtype=torch.float32)

    if return_only_node_features:
        return node_features

    # 2. 教師ターゲット（予測目標）データの結合パッキング処理
    if target is not None:
        if any(data.ndim > 1 for data in target.values()):
            target_features = np.hstack([data for data in target.values()])
            target_features = torch.tensor(target_features, dtype=torch.float32)
        else:
            target_features = np.vstack([data for data in target.values()]).T
            target_features = torch.tensor(target_features, dtype=torch.float32)
    else:
        target_features = None

    # 3. 要素接続情報（cells）のパースと、3D四面体から2D三角形ポリゴン面への展開幾何演算
    tetra = None
    cells = cells.T # [接続ノード数, セル数] の形状に変形
    cells = torch.tensor(cells)
    
    if cells.shape[0] == 4:
        # 行数が4の場合、3次元の四面体（tetrahedral）メッシュと判定
        tetra = cells
        # 四面体の周囲を囲む4つの三角形ポリゴン面（face）を、インデックスの組み合わせから全自動抽出して結合
        face = torch.cat(
            [
                cells[0:3], # 面 1: (0, 1, 2)
                cells[1:4], # 面 2: (1, 2, 3)
                torch.stack([cells[2], cells[3], cells[0]], dim=0), # 面 3: (2, 3, 0)
                torch.stack([cells[3], cells[0], cells[1]], dim=0), # 面 4: (3, 0, 1)
            ],
            dim=1,
        )
    if cells.shape[0] == 3:
        # 行数が3の場合、2次元の三角形（triangle）ポリゴンメッシュと判定
        face = cells

    # 完成した各テンソルをパッキングして PyG Data オブジェクトを新調
    return Data(
        x=node_features,
        face=face,
        tetra=tetra,
        y=target_features,
        pos=torch.tensor(points, dtype=torch.float32),
        id=id,
        next_data=next_data,
    )


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
) -> Data:
    """
    meshio.Mesh オブジェクトの幾何構造を直接パースし、
    内部の meshdata_to_graph を経由して PyG グラフデータへとコンバートする。
    """
    target = None
    # ターゲットメッシュと対象フィールド（流速等）が指定されている場合、教師ラベル配列を作成
    if target_mesh is not None and target_fields:
        target_data = [target_mesh.point_data[field] for field in target_fields]
        target = np.hstack(target_data)

    # メッシュ内のセルのうち、三角形（triangle）または四角形（quad）ポリゴン要素を垂直スタックして接続情報を抽出
    cells = np.vstack(
        [v for k, v in mesh.cells_dict.items() if k in ["triangle", "quad"]]
    )

    return meshdata_to_graph(
        points=mesh.points,
        cells=cells,
        point_data=mesh.point_data,
        time=time,
        target=target,
    )


def torch_graph_to_mesh(graph: Data, node_features_mapping: dict[str, int]) -> Mesh:
    """
    【逆変換エクスポート関数】
    モデルが予測・更新した PyG のグラフ構造（Data）をデタッチし、
    可視化用の meshio.Mesh オブジェクトへと復元逆コンバートする。
    """
    # 1. グラフのノード特徴量マトリクス（graph.x）の特定列から、指定された物理量（流速など）のNumPy配列を切り出す
    # ログを出力して、渡されるマッピング（キー名）を記録する
    try:
        logger.info(f"torch_graph_to_mesh: node_features_mapping keys: {list(node_features_mapping.keys())}")
    except Exception:
        pass
    point_data = {
        f: graph.x[:, indx].detach().cpu().numpy()
        for f, indx in node_features_mapping.items()
    }

    # 2. 座標次元数（2D平面か3D空間か）に応じて、復元するセルタイプ（triangle / tetra）を動的判定
    cells = graph.face.detach().cpu().numpy()
    if graph.pos.shape[1] == 2:
        extra_shape = 3
        cells_type = "triangle"
    elif graph.pos.shape[1] == 3:
        extra_shape = 4
        cells_type = "tetra"
    else:
        raise ValueError(f"座標次元数が不正です。2または3である必要があります。得られた次元: {graph.pos.shape[1]}")
        
    # 配列の並びが [接続ノード数, セル数] になっている場合は転置をとって [セル数, 接続ノード数] に直す
    if cells.shape[-1] != extra_shape:
        cells = cells.T

    # meshio の Mesh インスタンスを構築して返す（このあとエクスポートバイナリに書き出される）
    return Mesh(
        graph.pos.detach().cpu().numpy(),
        [(cells_type, cells)],
        point_data=point_data,
    )


def get_masked_indexes(graph: Data, masking_ratio: float = 0.15) -> torch.Tensor:
    """
    【MeshMask自己教師あり学習用】
    指定された隠蔽率（masking_ratio: デフォルト15%）に基づいて、
    メッシュグラフ全体のノード群から『生存（キープ）させるノードのIDインデックス配列』をランダムに選出する。
    """
    n, _ = graph.x.shape
    nodes_to_keep = 1 - masking_ratio      # 生存させるノードの割合（例: 85%）
    num_rows_to_sample = int(nodes_to_keep * n) # キープする具体的なノード個数
    
    # 0 〜 n-1 までの整数をランダムシャッフル（乱数パーミュテーション）
    random_indices = torch.randperm(n)
    # 先頭から必要個数分を切り出し（非復元ランダムサンプリング）
    selected_indices = random_indices[:num_rows_to_sample]

    return selected_indices
