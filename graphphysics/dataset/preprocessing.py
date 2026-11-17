import math
import random
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch_geometric.transforms as T
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from graphphysics.utils.nodetype import NodeType


def add_edge_features() -> List[Callable[[Data], Data]]:
    """
    グラフのエッジ（ノード間の繋がり）に対して、幾何学的な特徴量を追加するPyTorch Geometricの変換リストを返す。
    
    追加される特徴量:
    - Cartesian: デカルト座標における相対位置ベクトル (Δx, Δy, Δz)
    - Distance: ノード間のユークリッド距離 (||Δx, Δy, Δz||)
    """
    return [T.Cartesian(norm=False), T.Distance(norm=False)]


def _3d_face_to_edge(graph: Data) -> Data:
    """
    3Dの四角形ポリゴン（quadrilateral faces）を三角形の面として再構成し、エッジに変換する。
    
    パラメータ:
        graph (Data): 入力グラフデータ
    """
    face = graph.face
    # 四角形(0,1,2,3)を2つの三角形(0,1,2 と 0,2,3 などを考慮)に分割し、
    # さらに隣接関係を網羅するように再構築する
    graph.face = torch.cat(
        [
            face[0:3],
            face[1:4],
            torch.stack([face[2], face[3], face[0]], dim=0),
            torch.stack([face[3], face[0], face[1]], dim=0),
        ],
        dim=1,
    )
    return graph


def add_obstacles_next_pos(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
) -> Data:
    """
    障害物（OBSTACLE）が動くシミュレーションにおいて、次の時刻への変位量（移動ベクトル）を計算し、
    それをノード特徴量として追加する。
    """
    # 現在のワールド座標（位置）と、それ以外の特徴量を分離
    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    other_features = graph.x[:, world_pos_index_end:]

    # 教師データ(graph.y)から、次の時刻のターゲットとなるワールド座標を取得
    target_world_pos = graph.y[:, world_pos_index_start:world_pos_index_end]

    # 障害物がどれだけ動くか（変位）を計算
    obstacle_displacement = target_world_pos - world_pos

    # ノードのタイプ（流体、壁、障害物など）を取得
    # (-3 しているのは、この関数の最後に変位(3次元)を結合することでインデックスがズレることを考慮しているため)
    node_type = graph.x[:, node_type_index - 3]

    # 障害物(OBSTACLE)以外のノードに対しては、変位を障害物全体の平均値で埋める（あるいは0にリセットする用途）
    only_obstacle_displacement = obstacle_displacement[node_type == NodeType.OBSTACLE]
    mean_obstacle_displacement = torch.mean(only_obstacle_displacement, dim=0)
    obstacle_displacement[node_type != NodeType.OBSTACLE] = mean_obstacle_displacement

    # 元の特徴量配列の間に変位量（obstacle_displacement）を挟み込んで更新
    graph.x = torch.cat([world_pos, obstacle_displacement, other_features], dim=1)
    return graph


def add_world_edges(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
    radius: float = 0.03,
) -> Data:
    """
    メッシュのトポロジー（本来の繋がり）にはないが、空間的に近くにあるノード同士を強制的に結ぶ
    「ワールドエッジ」を追加する。動く障害物と流体の相互作用などを学習しやすくなる。
    """

    # KDTreeを使って、指定した半径(radius)内にあるノードのペアを高速に検索する内部関数
    def _close_pairs_ckdtree(X, max_d):
        tree = cKDTree(X.cpu().numpy())
        pairs = tree.query_pairs(max_d, output_type="ndarray")
        return torch.Tensor(pairs.T).long()

    # ワールド座標を取得し、近いノードペアを計算
    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    added_edges = _close_pairs_ckdtree(world_pos, radius).to(graph.x.device)

    type = graph.x[:, node_type_index]

    # 追加するエッジは、「障害物(OBSTACLE)」と「通常の流体(NORMAL)」の間に限定するフィルタリング
    m1 = torch.gather(type, -1, added_edges[0]) == NodeType.OBSTACLE
    m2 = torch.gather(type, -1, added_edges[1]) == NodeType.NORMAL
    mask1 = torch.logical_and(m1, m2)

    m1 = torch.gather(type, -1, added_edges[0]) == NodeType.NORMAL
    m2 = torch.gather(type, -1, added_edges[1]) == NodeType.OBSTACLE
    mask2 = torch.logical_and(m1, m2)

    mask = torch.logical_or(mask1, mask2)
    added_edges = added_edges[:, mask]

    # 元のメッシュエッジに、新しく見つけた空間エッジを結合し、無向グラフ（双方向）化する
    edge_index = torch.cat([added_edges, graph.edge_index], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

    graph.edge_index = edge_index
    return graph


def add_world_pos_features(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
) -> Data:
    """
    ワールド座標（絶対座標）の情報を基に、エッジの属性（edge_attr）として
    相対的な位置ベクトルと距離を計算して追加する。
    """
    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    senders, receivers = graph.edge_index

    # エッジの始点と終点の相対位置（ベクトル）と、そのノルム（距離）を計算
    relative_world_pos = world_pos[senders] - world_pos[receivers]
    relative_world_pos_norm = torch.norm(relative_world_pos, p=2, dim=-1, keepdim=True)

    # 既存のエッジ特徴量に結合
    graph.edge_attr = torch.cat(
        [
            graph.edge_attr,
            relative_world_pos.type_as(graph.edge_attr),
            relative_world_pos_norm.type_as(graph.edge_attr),
        ],
        dim=-1,
    )

    return graph


def add_noise(
    graph: Data,
    noise_index_start: Union[int, List[int]],
    noise_index_end: Union[int, List[int]],
    noise_scale: Union[float, List[float]],
    node_type_index: int,
    t: Optional[float] = None,
) -> Data:
    """
    長期間の自律回帰予測（Rollout）で誤差が蓄積して崩壊するのを防ぐため、
    訓練時の入力特徴量（速度など）にあえてガウスノイズを加え、ノイズから元の状態を復元する堅牢性（ロバスト性）を学習させる。
    """
    # インデックス指定がリストでない場合はリストに変換
    if isinstance(noise_index_start, int):
        noise_index_start = [noise_index_start]
    if isinstance(noise_index_end, int):
        noise_index_end = [noise_index_end]

    # ノイズのスケール（標準偏差）指定が単一のfloatの場合は、インデックスの数に合わせてリスト化
    if isinstance(noise_scale, float):
        noise_scale = [noise_scale] * len(noise_index_start)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError("noise_index_start と noise_index_end の長さは同じである必要があります。")
    if len(noise_scale) != len(noise_index_start):
        raise ValueError("noise_scale はインデックスリストと同じ長さである必要があります。")

    node_type = graph.x[:, node_type_index]

    # 壁面や流入・流出などの境界条件（NORMAL以外）にはノイズを乗せないようにするためのマスク
    mask = node_type != NodeType.NORMAL

    for start, end, scale in zip(noise_index_start, noise_index_end, noise_scale):
        feature = graph.x[:, start:end]

        # tが指定されている場合、カリキュラム学習（学習が進むにつれてノイズ量を変化させる）を適用
        scale_ = 10 * scale * (1 + math.cos(t * math.pi)) if t is not None else scale

        # ガウスノイズを生成
        noise = torch.randn_like(feature) * scale_

        # NORMAL（流体）以外のノードはノイズを0にして影響を消す
        noise[mask] = 0

        # 元の特徴量にノイズを加算して更新
        graph.x[:, start:end] = feature + noise

    return graph


def compute_min_distance_to_type(
    graph: Data, target_type: NodeType, node_types: torch.Tensor
):
    """
    各ノードから、指定されたタイプ（例：壁）の最も近いノードまでの最小距離を計算する。
    （壁からの距離、といった物理的に意味のある特徴量を作るのに役立つ）
    """
    # ターゲットタイプ（例：壁）に該当するノードのマスクを取得
    type_a_mask = node_types == target_type

    # 全ノードの3次元座標を取得 [num_nodes, 3]
    pos = graph.pos

    # PyTorchのブロードキャスト機能を使って総当たりで距離を計算するため次元を拡張
    # pos_expanded: [num_nodes, 1, 3], pos_type_a: [1, num_type_a_nodes, 3]
    pos_expanded = pos.unsqueeze(1)
    pos_type_a = pos[type_a_mask].unsqueeze(0)

    # ターゲット群のすべてのノードとの距離を計算
    distances = torch.sqrt(torch.sum((pos_expanded - pos_type_a) ** 2, dim=-1))

    # 各ノードごとに、ターゲットノード群の中で一番近いもの（最小距離）を取得
    min_distances = torch.min(distances, dim=1)[0]

    return min_distances


class Random3DRotate(BaseTransform):
    """
    データ拡張（Data Augmentation）用クラス。
    グラフの座標（pos）やベクトル特徴量（速度など）全体を、ランダムな3D角度で回転させることで、
    モデルが特定の向きに過学習するのを防ぐ。
    """
    def __init__(self, feature_indices: List[Tuple[int, int]] = None) -> None:
        self.feature_indices = feature_indices or []
        # 回転させる特徴量は、必ず x, y, z の3次元（要素数が3）でなければならない
        for start_idx, end_idx in self.feature_indices:
            assert end_idx - start_idx == 3, (
                f"回転対象の特徴量は3つの要素(xyz)を持つ必要があります。 指定された範囲: {start_idx}-{end_idx}"
            )

    def _get_random_angles(self):
        """yaw, pitch, roll に相当する3軸のランダムな回転角（ラジアン）を生成"""
        angles = [random.uniform(-180, 180) for _ in range(3)]
        return [math.radians(angle) for angle in angles]

    def _build_rotation_matrix(self, alpha, beta, gamma):
        """3つの角度から 3x3 の回転行列（Rotation Matrix）を構築する"""
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        cos_b, sin_b = math.cos(beta), math.sin(beta)
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)

        # 3次元の回転行列の公式
        matrix = [
            [
                cos_a * cos_b,
                cos_a * sin_b * sin_g + sin_a * cos_g,
                -cos_a * sin_b * cos_g + sin_a * sin_g,
            ],
            [
                -sin_a * cos_b,
                -sin_a * sin_b * sin_g + cos_a * cos_g,
                sin_a * sin_b * cos_g + cos_a * sin_g,
            ],
            [sin_b, -cos_b * sin_g, cos_b * cos_g],
        ]

        return torch.tensor(matrix)

    def _rotate_features(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """特徴量（速度ベクトルなど）に回転行列を掛けて座標変換する"""
        for start_idx, end_idx in self.feature_indices:
            feat = x[:, start_idx:end_idx]
            rotated_feat = feat @ matrix.to(feat.device, feat.dtype)
            x[:, start_idx:end_idx] = rotated_feat
        return x

    def forward(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        # ランダムな角度を生成し、回転行列を作る
        alpha, beta, gamma = self._get_random_angles()
        rotation_matrix = self._build_rotation_matrix(alpha, beta, gamma)

        # メッシュのノード座標(pos)を回転
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
            assert pos.size(-1) == 3, "Node positions must be 3-dimensional"
            data.pos = pos @ rotation_matrix.to(pos.device, pos.dtype)

        # 指定された特徴量配列(速度ベクトルなど)を回転
        if hasattr(data, "x") and data.x is not None and self.feature_indices:
            data.x = self._rotate_features(data.x, rotation_matrix)

        # 教師ラベル(ターゲット予測値)のベクトルも整合性がとれるように回転
        if hasattr(data, "y") and data.x is not None:
            target = data.y[:, 0:3]
            rotated_target = target @ rotation_matrix.to(target.device, target.dtype)
            data.y = rotated_target

        return data


def build_preprocessing(
    noise_parameters: Optional[dict] = None,
    world_pos_parameters: Optional[dict] = None,
    add_edges_features: bool = True,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
) -> T.Compose:
    """
    設定JSONのパラメータに基づいて、これまで定義した各前処理関数を結合し、
    一連のパイプライン（T.Compose）として構築するファクトリ関数。
    """
    preprocessing: List[Callable[[Data], Data]] = []

    # 1. ユーザー定義のカスタムノード特徴量付加処理を最初に追加
    if extra_node_features is not None:
        if not isinstance(extra_node_features, list):
            extra_node_features = [extra_node_features]
        preprocessing.extend(extra_node_features)

    # 2. ワールド座標関連の処理（動く障害物に対するワールドエッジの追加など）
    if world_pos_parameters is not None:
        preprocessing.extend(
            [
                partial(
                    add_obstacles_next_pos,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                ),
                T.FaceToEdge(remove_faces=False), # ポリゴン面をエッジに変換
                partial(
                    add_world_edges,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                    radius=world_pos_parameters.get("radius", 0.03),
                ),
            ]
        )
        # エッジの長さや相対位置を追加
        preprocessing.extend(add_edge_features())
    else:
        # ワールド座標を扱わない通常の静的メッシュ処理
        preprocessing.append(T.FaceToEdge(remove_faces=False))
        if add_edges_features:
            preprocessing.extend(add_edge_features())

    # 3. ノイズ付加処理（リストの先頭寄り、基本処理の直後に挿入）
    if noise_parameters is not None:
        add_noise_transform = partial(
            add_noise,
            noise_index_start=noise_parameters["noise_index_start"],
            noise_index_end=noise_parameters["noise_index_end"],
            noise_scale=noise_parameters["noise_scale"],
            node_type_index=noise_parameters["node_type_index"],
        )
        preprocessing.insert(1, add_noise_transform)

    # 4. 最後に実行されるべきユーザー定義のカスタムエッジ特徴量処理を追加
    if extra_edge_features is not None:
        if not isinstance(extra_edge_features, list):
            extra_edge_features = [extra_edge_features]
        preprocessing.extend(extra_edge_features)

    # T.Compose でラップすることで、DataLoader側で data = preprocessing(data) と一括処理できるようにする
    return T.Compose(preprocessing)