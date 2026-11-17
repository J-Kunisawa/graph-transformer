import torch
from torch_geometric.data import Data

def build_features(graph: Data) -> Data:
    # 1. 最初の3次元（速度2D + 圧力1D）を現在の物理量として抽出
    current_velocity_and_p = graph.x[:, 0:3]
    
    # 2. 4列目（インデックス3以降）に存在する、元々の境界条件データをすべて退避させておく
    # (ワンホット前なら1次元、もしすでにワンホット化されているシステムなら複数列を丸ごとキープ)
    original_node_type = graph.x[:, 3:]
    
    # 3. 過去データの存在チェックと抽出
    if getattr(graph, "previous_data", None) is not None:
        # H5データセットが確保してくれた過去の物理量（0〜2列目）を取得
        previous_data = graph.previous_data[:, 0:3].to(current_velocity_and_p.device)
    else:
        # 初期フレーム（t=0）用の安全策：現時刻のコピー
        previous_data = current_velocity_and_p.clone()
        
    # 4. 物理的な時間発展の差分（加速度・圧力変化率）を計算
    diff = current_velocity_and_p - previous_data
    
    # 5. 【超重要】物理特徴量（現3 + 差分3 = 計6次元）を左側に固め、
    # その後ろに退避させておいた境界条件データをそのまま結合して上書きする
    graph.x = torch.cat([current_velocity_and_p, diff, original_node_type], dim=1)
    
    return graph