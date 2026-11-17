import os, json, torch
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
from graphphysics.training.parse_parameters import get_dataset, get_preprocessing
from graphphysics.training.lightning_module import LightningModule
from torch_geometric.loader import DataLoader

# ====== 入力 ======
CFG = "training_config/cylinder_local.json"   # あなたの設定JSON
CKPT = "checkpoints/epoch=9-step=302500-v1.ckpt" # 学習済みckpt（適宜変更）
OUT  = "viz_pred.png"                         # 出力画像

# ====== 準備 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42, workers=True)

with open(CFG, "r") as fp:
    params = json.load(fp)

# 既存の学習と同じ前処理
preproc = get_preprocessing(param=params, device=device, use_edge_feature=False)
val_ds  = get_dataset(param=params, preprocessing=preproc, use_edge_feature=False, use_previous_data=False, switch_to_val=True)

# 検証サンプルを1件取り出す（バッチ化しない）
loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# LightningModule をロード
model = LightningModule.load_from_checkpoint(
    checkpoint_path=CKPT,
    parameters=params,
    learning_rate=1e-3, warmup=0, num_steps=1,
    trajectory_length=val_ds.trajectory_length,
    timestep=val_ds.dt,
    use_previous_data=False
).to(device)
model.eval()

# ====== 推論 ======
batch = next(iter(loader)).to(device)

# Simulator.forward は (graph, target_delta_normalized) を内部で組み、
# 最終的に (node_embeddings, edges, predicted_outputs) を返す設計
with torch.no_grad():
    _, _, pred_delta = model.model(batch)   # Δ(速度) を予測
    # 出力正規化を内部で戻していれば pred_delta は物理スケール
    # もし Δ ではなく次時刻そのものが欲しければ current + Δ
    current = batch.y if hasattr(batch, "y") else batch.target
    pred = current + pred_delta

# ====== 可視化用テンソル取り出し ======
# 位置（2次元）
pos = batch.pos.detach().cpu().numpy()        # [N, 2]
# 真値速度（次時刻）
gt  = (batch.y if hasattr(batch, "y") else batch.target).detach().cpu().numpy()  # [N, 2]
# 予測速度（次時刻）
pred = pred.detach().cpu().numpy()
# 誤差ノルム
err = ((pred - gt) ** 2).sum(axis=1) ** 0.5

# ====== 描画 ======
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# 1) Ground Truth
axes[0].quiver(pos[:,0], pos[:,1], gt[:,0], gt[:,1], angles='xy', scale_units='xy', scale=None, linewidths=0.3)
axes[0].set_title("Ground Truth velocity")
axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])

# 2) Prediction
axes[1].quiver(pos[:,0], pos[:,1], pred[:,0], pred[:,1], angles='xy', scale_units='xy', scale=None, linewidths=0.3)
axes[1].set_title("Predicted velocity")
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])

# 3) Error magnitude
im = axes[2].scatter(pos[:,0], pos[:,1], c=err, s=3)
axes[2].set_title("Error |u_pred - u_gt|")
axes[2].set_aspect("equal"); axes[2].set_xticks([]); axes[2].set_yticks([])
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="speed error")

plt.savefig(OUT, dpi=180)
print(f"Saved: {OUT}")
