import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

# 演算を実行するデバイスの決定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gradient_weighted_least_squares(
    graph: Data, field: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    【最小二乗法勾配スキーム（商用CFDソルバー・VTK互換）】
    
    不規則に歪んだ非構造格子上でも、周囲の幾何構造を反映して最も数学的確実性の高い
    空間勾配（Gradient：∇u）を算出する。セル中心の勾配を求めた後、体積重み付きでノードへ集約する。
    
    戻り値:
        gradients: 形状 [ノード数, 物理次元F, 空間次元D] の勾配テンソル
    """

    # 1. デバイス転送と次元の整形
    points = graph.pos.to(device)  # 座標テンソル: (N, 2) または (N, 3)
    field = field.to(device)       # 物理場テンソル: (N, F)

    if field.ndim == 1:
        field = field.unsqueeze(1) # スカラー場の場合、2次元テンソル (N, 1) に拡張

    dim_x = points.shape[1] # 空間次元数 (2 または 3)
    dim_u = field.shape[1]  # 物理量の次元数

    # 要素（セル）のトポロジー接続情報を抽出 [要素数M, 頂点数D+1]
    elements = graph.face.T.to(device)  

    D = elements.shape[1] - 1  # 2次元（三角形: 3頂点）なら D=2、3次元（四面体: 4頂点）なら D=3
    N = points.shape[0]

    # 2. セルを構成する各頂点の「絶対座標」と「物理量」を一括参照（Gather）
    elem_points = points[elements] # 形状: (M, D+1, dim_x)
    elem_field = field[elements]   # 形状: (M, D+1, dim_u)

    # 各要素の「第0頂点」を原点（基準）とした、周辺頂点の幾何相対ベクトル（A）と物理量差分（B）を算出
    A = elem_points[:, 1:, :] - elem_points[:, 0:1, :]  # 行列 A: (M, D, dim_x)
    B = elem_field[:, 1:, :] - elem_field[:, 0:1, :]    # 行列 B: (M, D, dim_u)

    # 3. 線形最小二乗方程式  A @ grad^T = B  を解く
    # 各セルの形状変化を反映した、セル中心における不連続勾配テンソルを取得
    # grad_elems 形状: (M, dim_u, dim_x)
    grad_elems = torch.linalg.lstsq(A, B).solution.transpose(1, 2)

    # 4. セルのメジャー（2Dなら三角形の面積、3Dなら四面体の体積）を幾何計算
    if D == 2:  # 三角形セルの場合
        v1 = A[:, 0, :]  # 辺ベクトル 1
        v2 = A[:, 1, :]  # 辺ベクトル 2
        if v1.shape[1] == 3:
            # 3次元空間内の2D表面要素（外積ベクトルのノルムの 0.5 倍）
            cross = torch.cross(v1, v2, dim=1)  
            volume = 0.5 * torch.norm(cross, dim=1)  
        if v1.shape[1] == 2:
            # 純粋な2D平面上の三角形の面積（行列式の絶対値の 0.5 倍）
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            volume = 0.5 * torch.abs(cross)
            
    elif D == 3:  # 四面体セルの場合
        # 3D空間ベクトルの行列式の絶対値を 6.0 で割ることで四面体の体積を算出
        volume = torch.abs(torch.linalg.det(A)) / 6.0  
    else:
        raise ValueError(f"サポートされていない要素次元数です: D={D}")

    # 5. セル中心の勾配値を、セルの体積（面積）を重みとして担当頂点ノードへと集約（Scatter Add）
    gradients = torch.zeros((N, dim_u, dim_x), device=device)
    weights = torch.zeros((N, 1), device=device)

    for i in range(D + 1):
        idx = elements[:, i] # 各頂点のグローバルノードIDを取得
        # 体積を掛け合わせた勾配の総和をノードへ蓄積
        gradients.index_add_(0, idx, grad_elems * volume[:, None, None])
        # 分母となる、各ノードの周囲を囲むセルの総体積（重み総和）を蓄積
        weights.index_add_(0, idx, volume[:, None])

    # 最終的に総体積で除算してノード上の連続な勾配場を確定（ゼロ除算は clamp で防止）
    gradients /= weights.clamp(min=1e-12).view(-1, 1, 1)

    return gradients


def compute_gradient_finite_differences(
    graph: Data, field: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    【重み付きグラフ有限差分勾配スキーム】
    
    メッシュのトポロジーエッジ（繋がり）に沿って、両端ノード間の距離の逆数を重みとした
    有限差分（du/dx）を計算し、ノード上で集約・平均化する。
    """
    pos = graph.pos.to(device)
    edges = graph.edge_index.to(device)
    
    # メッセージパッシングの重複を無くすため、エッジをソートしてユニークな無向リンクのみに絞り込み
    edges = torch.unique(torch.sort(edges.T, dim=1)[0], dim=0).T
    field = field.to(device)

    N, D = pos.shape
    _, F = field.shape
    i, j = edges[0], edges[1] # エッジの始点ノードID（i）と終点ノードID（j）

    eps = 1e-8

    # グラフエッジ間の相対距離ベクトル（dx）と物理量差分（du）の計算
    dx = pos[j] - pos[i]
    du = field[j] - field[i]
    distances = torch.norm(dx, dim=1) # 格子点間のユークリッド距離

    # エッジ上での勾配寄与量 (du ⊗ dx) / (||dx||^2) を外積テンソルとして計算
    gradient_edges = torch.matmul(du.unsqueeze(2), dx.unsqueeze(1)) / (
        distances.view(-1, 1, 1) ** 2 + eps
    )

    # 距離の2乗の逆数（1 / r^2）を、空間的に近い格子ほど微分寄与度を高くするための重みとして定義
    weight_edges = 1.0 / (distances**2 + eps)
    weight_sums = torch.zeros((N, F, D), device=device)
    
    # 始点と終点の双方のノードバッファへ重み値を加算
    weight_sums.index_add_(0, i, weight_edges.view(-1, 1, 1).expand(-1, F, D))
    weight_sums.index_add_(0, j, weight_edges.view(-1, 1, 1).expand(-1, F, D))

    # エッジ勾配に重みを乗算し、両端のノードへ集約
    gradient_edges = gradient_edges * weight_edges.view(-1, 1, 1)
    gradient = torch.zeros((N, F, D), device=device)
    gradient.index_add_(0, i, gradient_edges)
    gradient.index_add_(0, j, gradient_edges)

    # 各ノードに集まった総重みで割ることで、近傍距離を考慮した有限差分勾配を確定
    gradient = gradient / (weight_sums + eps)
    return gradient


def compute_gradient(
    graph: Data,
    field: torch.Tensor,
    method: str = "least_squares",
    device: str = "cpu",
) -> torch.Tensor:
    """
    設定された文字列方法（least_squares または finite_diff）に応じて、
    不規則メッシュ上の勾配計算を適切なアルゴリズムへ動的に分岐マッピングするエントリポイント。
    """
    if method == "least_squares":
        return compute_gradient_weighted_least_squares(graph, field, device=device)
    elif method == "finite_diff":
        return compute_gradient_finite_differences(graph, field, device=device)
    else:
        raise ValueError(f"未知の勾配計算手法が指定されました: {method}")


def compute_vector_gradient_product(
    graph: Data,
    field: torch.Tensor,
    gradient: torch.Tensor = None,
    method: str = "finite_diff",
    device: str = "cpu",
) -> torch.Tensor:
    """
    【物理情報：流体対流項 (u・∇)u 演算】
    
    ベクトル場（field: 流速 u）と、その空間勾配テンソル（gradient: ∇u）の点ごとの積（内積総和）を計算する。
    ナビエ・ストークス方程式の非線形対流項をロス関数として課すための必須関数。
    """
    field = field.to(device)
    
    # 勾配がまだ事前計算されていない場合は、ここで動的に構築
    if gradient is None:
        gradient = compute_gradient(
            graph, field, method=method, device=device
        )  # 算出した形状: (N, F, D)
    else:
        gradient.to(device)

    # アインシュタイン縮約記法（einsum）による超高速内積演算
    # ノード n、物理成分 f、空間軸 d に対し、d 軸（XYZ成分）の次元に沿って要素積をとり総和（縮約）する。
    # 物理的な意味:  u_x*(∂u/∂x) + u_y*(∂u/∂y) + u_z*(∂u/∂z)
    product = torch.einsum(
        "nf,nfd->nf", field, gradient
    )  
    return product


def compute_divergence(
    graph: Data,
    field: torch.Tensor,
    gradient: torch.Tensor = None,
    method: str = "finite_diff",
    device: str = "cpu",
) -> torch.Tensor:
    """
    【物理情報：質量保存則・連続の式 ∇・u 演算】
    
    非構造グラフ格子上のベクトル場（field）の「発散（Divergence：∇・u）」を計算する。
    非圧縮性流体（質量保存則）の制約 ∇・u = 0 を満たしているかの評価指標・ペナルティとして使用。
    """
    # 勾配テンソル (N, F, D) が無い場合は計算
    if gradient is None:
        gradient = compute_gradient(
            graph, field, method=method, device=device
        )  # 形状: (N, F, D)

    # 空間軸と物理成分が一致する成分（∂u_x/∂x, ∂u_y/∂y, ∂u_z/∂z）マトリクスの対角成分（diagonal）を切り出し、
    # 最終次元（dim=-1）に沿って総和（sum）をとることで、各ノードにおける発散スカラー値を確定。
    divergence = gradient.diagonal(dim1=1, dim2=2).sum(dim=-1) # 形状: (N,)
    return divergence