import json
import torch
from torch_geometric.loader import DataLoader
from graphphysics.training.parse_parameters import get_preprocessing, get_dataset, get_model, get_simulator
from graphphysics.external.aneurysm import build_features

PARAM_PATH = "training_config/openfoam_gnn.json"

def main():
    with open(PARAM_PATH, 'r') as f:
        params = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 本番設定では edge_input_size>0 のためエッジ特徴量を有効化する
    pre = get_preprocessing(param=params, device=device, use_edge_feature=True, extra_node_features=build_features)
    ds = get_dataset(param=params, preprocessing=pre, use_edge_feature=True)
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    model = get_model(param=params)
    sim = get_simulator(param=params, model=model, device=device)
    it = iter(dl)
    batch = next(it)
    batch = batch.to(device)
    net_out, tgt_norm, outputs = sim(batch)
    print('network_output shape:', net_out.shape)
    print('target_norm shape:', tgt_norm.shape)
    print('network_output stats:', net_out.min().item(), net_out.max().item())
    print('target_norm stats:', tgt_norm.min().item(), tgt_norm.max().item())
    # 追加診断: pre_target, target_delta (正規化前), node features, normalizer 状態
    pre_target = sim._get_pre_target(batch)
    target_delta = batch.y - pre_target
    print('pre_target stats:', pre_target.min().item(), pre_target.max().item())
    print('target_delta (raw) stats:', target_delta.min().item(), target_delta.max().item())
    # node features（正規化前）
    one_hot = sim._get_one_hot_type(batch)
    node_features = sim._build_node_features(batch, one_hot)
    print('node_features (raw) stats:', node_features.min().item(), node_features.max().item())
    # 正規化器の内部状態を表示
    try:
        out_vars = sim._output_normalizer.get_variable()
        node_vars = sim._node_normalizer.get_variable()
        print('output_normalizer:', out_vars)
        print('node_normalizer:', node_vars)
    except Exception as e:
        print('Failed to get normalizer variables:', e)

if __name__ == '__main__':
    main()
