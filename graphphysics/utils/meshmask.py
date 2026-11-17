import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

# 演算を実行するデバイス（GPU/CPU）の自動決定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_edges(
    edge_index: torch.Tensor, node_index: torch.Tensor, edge_attr: torch.Tensor = None
):
    """
    指定された生存ノードID（node_index）に基づいてグラフのエッジをフィルタリングし、
    生存したコンパクトなノード番号に合わせてエッジインデックスを 0 から振り直す（再インデックス）。

    パラメータ:
        edge_index (Tensor): 元のグラフのエッジトポロジー [2, NumEdges]
        node_index (Tensor): マスキングを免れて生存（選択）させるノードのID群
        edge_attr (Tensor, optional): 各エッジに紐付く幾何特徴量
    """
    # グラフの総ノード数を動的に取得
    num_nodes = maybe_num_nodes(edge_index, None)
    node_index = node_index.to(device)
    
    # 切り出された新しい部分グラフ内での連番インデックス（0, 1, 2, ...）を作成
    cluster_index = torch.arange(node_index.size(0), device=node_index.device)

    # ループ無しの超高速再インデックス用ルックアップテーブル（マスク配列）の構築
    # 全ノード分の配列を一度 -1 で満たし、生存ノードの場所にだけ新しい連番インデックスを代入
    mask = node_index.new_full((num_nodes,), -1).to(device)
    mask[node_index] = cluster_index

    # 全エッジの始点（senders）と終点（receivers）を取得
    senders, receivers = edge_index[0], edge_index[1]
    
    # ルックアップテーブルを参照し、古いノードIDを新しい連番IDへ一括置換
    senders, receivers = mask[senders], mask[receivers]
    
    # 始点と終点の双方が生存（インデックスが0以上）しているエッジのみを True とするマスクを作成
    mask = (senders >= 0) & (receivers >= 0)
    
    # 生存したエッジだけを切り出し
    senders, receivers = senders[mask], receivers[mask]

    # エッジ特徴量（距離など）が存在する場合は、同様にフィルタリング
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    # 再インデックスされた新しいエッジ構造、エッジ特徴量、および生存エッジマスクを返す
    return torch.stack([senders, receivers], dim=0), edge_attr, mask


def build_masked_graph(
    masked_graph: Data,
    selected_indexes: torch.Tensor,
):
    """
    Masked Autoencoder（MAE）タスクの順伝播用関数。
    入力されたメッシュグラフから、指定した生存インデックス（selected_indexes）以外のノードを隠蔽し、
    ノード特徴量、座標、トポロジーがすべてコンパクトにクリップされた『部分グラフ』を生成する。
    """
    # 1. filter_edges を呼び出し、トポロジーエッジの切り出しと再インデックス化を実行
    masked_e_index, masked_e_attr, edges_mask = filter_edges(
        masked_graph.edge_index, selected_indexes, masked_graph.edge_attr
    )

    # 2. グラフオブジェクトの各属性を選択された生存領域だけに破壊的更新（上書き）
    masked_graph.edge_index = masked_e_index
    masked_graph.x = masked_graph.x[selected_indexes] # 生存ノードの特徴量のみをキープ
    
    if masked_graph.pos is not None:
        masked_graph.pos = masked_graph.pos[selected_indexes] # 生存ノードの座標のみをキープ

    if masked_e_attr is not None:
        masked_graph.edge_attr = masked_e_attr

    return masked_graph, edges_mask


def reconstruct_graph(
    graph: Data,
    latent_masked_graph: Data,
    selected_indexes: torch.Tensor,
    node_mask_token: torch.nn.Parameter,
    edges_mask: torch.Tensor,
    edge_encoder: nn.Module = None,
    edge_mask_token: torch.nn.Parameter = None,
) -> Data:
    """
    Masked Decoder（復元デコーダ）の直前で呼び出されるグラフ再構築関数。
    
    部分グラフから計算された潜在特徴量（x）を、元々の広大なメッシュの正しい配置（selected_indexes）へ書き戻し、
    あらかじめ隠蔽（マスク）されていた領域には、学習可能な共有パラメータである『 [MASK] トークン 』を敷き詰めて
    元の完全なメッシュ構造へと復元統合する。
    """
    # 元のメッシュのノード数（n）と潜在次元数（f）を取得
    n, f = graph.x.shape
    
    # 1. 全ノードを一度、学習可能な [MASK] トークンベクトルのコピーで初期充填する
    features = torch.zeros(n, f).to(device)
    features += node_mask_token.expand(n, -1)
    
    # 2. マスクを免れていた領域（selected_indexes）に対して、部分グラフが高度に計算した潜在特徴量を上書きマッピング
    features[selected_indexes] = latent_masked_graph.x

    # 元のグラフのトポロジーをクローンし、今パッキングした特徴量を割り当て
    latent_graph = graph.clone()
    latent_graph.x = features

    # エッジ特徴量に対しても、全く同様の [MASK] トークン埋め込みと部分グラフからの復元を実行
    if graph.edge_attr is not None:
        n_edges, _ = graph.edge_attr.shape
        # 生のエッジ特徴量をエンコーダで潜在空間へ写像
        latent_graph.edge_attr = edge_encoder(latent_graph.edge_attr)
        # 一度すべてをエッジ用の [MASK] トークンで満たす
        latent_graph.edge_attr += edge_mask_token.expand(n_edges, -1)
        # 生存していたエッジ領域にだけ、部分グラフ側で計算していたエッジ特徴量を復元マッピング
        latent_graph.edge_attr[edges_mask] = latent_masked_graph.edge_attr

    return latent_graph