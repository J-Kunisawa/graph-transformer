# graph-transformer

非構造メッシュ上の物理シミュレーション（流体力学）を Graph Neural Network で学習する研究用リポジトリ。
円柱まわりの流れを対象に、Graph Transformer（Garnier et al., 2025）と MeshGraphNet 系 GNN を同一のパイプラインで比較できるように構築している。

本リポジトリは [DonsetPG/graph-physics](https://github.com/DonsetPG/graph-physics) をベースに、以下の拡張を加えたものである。

## 特徴（ベース実装からの拡張）

- **Adjacency Augmentation**: Graph Transformer の受容野を広げる仕組みを `model.adjacency_augmentation` で設定可能にした
  - Random edges（ランダム辺の追加）
  - Dilation（layer・head ごとに異なる k-hop 隣接行列を割り当てる Dilated Attention。ちょうど k 歩の辺 A^k を使用）
  - Global Attention（指定した node type（例: OBSTACLE）のノードへの大域辺をサンプリングして追加）
- **EAGLE の追加**: Janny et al. (ICLR 2023) の EAGLE を比較モデルとして実装（`graphphysics/models/eagle.py`、クラスタがない場合は Morton 順序で代替）
- **Multi-step loss + Scheduled Sampling**: 連続する複数ステップを自分の予測で回しながら学習し、長期ロールアウトの誤差蓄積に強くした
- **prev 特徴量**: 現在と1ステップ前の差分（加速度・圧力変化率に相当）を入力特徴量に追加（`graphphysics/external/append_previous.py`）
- **自前 OpenFOAM データセット**: OpenFOAM の計算結果を H5 / XDMF 形式へ変換するスクリプト群を用意し、Reynolds 数を変えた学習・テスト（学習 Re 100–300、テスト Re 80–400）に対応
- **実験スイープと評価**: モデル × 深さ × メッシュ解像度の一括実験、解像度別 RMSE・推論時間のプロット、FFT による Strouhal 数の推定スクリプト

## モデル

設定 JSON の `model.type` で切り替える。

| type | モデル | 備考 |
|---|---|---|
| `epd` | MeshGraphNet 系（Encode-Process-Decode） | 辺特徴あり |
| `transformer` | Graph Transformer（Garnier et al., 2025） | 辺特徴なし、隣接行列を attention マスクとして使用 |
| `transolver` | Transolver++ | 比較用ラッパー |
| `eagle` | EAGLE（Janny et al., 2023） | クラスタ pooling + global attention |

予測は絶対値ではなく1ステップの変化量（Δ）を学習し、検証・推論では自己回帰ロールアウトで評価する。
円柱の標準設定では速度と圧力を同時に出力する（output 3次元）。

## リポジトリ構成

```
graphphysics/     本体パッケージ（dataset / models / training / utils / external）
training_config/  学習設定 JSON（モデル・データセット・損失などの組み合わせ）
dataset_config/   データセット定義（meta JSON）
scripts/          データ変換・実験スイープ・評価・可視化スクリプト
train.py          学習の入口
predict.py        推論（ロールアウト予測）の入口
run_train.sh      学習実行スクリプト
run_predict.sh    推論実行スクリプト
Dockerfile        CUDA 12.4 + PyTorch 2.4 + DGL + PyG の実行環境
```

## セットアップ

### Docker（推奨）

```bash
docker compose build
docker compose up -d
docker compose exec graph-physics bash
```

### pip

Python 3.10 系を想定。PyTorch / DGL / PyTorch Geometric は CUDA 環境に合わせて公式 wheel から入れること（手順は `Dockerfile` を参照）。

```bash
pip install -r requirements.txt
```

学習のログ管理に Weights & Biases を使うため、事前に `wandb login` が必要。

## 使い方

データセット（H5 または XDMF）は設定 JSON 内のパス（例: `data/...`）に配置する。リポジトリには含まれない。

### 学習

```bash
bash run_train.sh
# または直接
python train.py \
  --training_parameters_path=training_config/xdmf_transformer.json \
  --num_epochs=150 \
  --init_lr=0.001 \
  --batch_size=4 \
  --model_save_name=model
```

### 推論

```bash
bash run_predict.sh
# または直接
python predict.py \
  --predict_parameters_path=training_config/xdmf_transformer.json \
  --model_path=checkpoints/model.ckpt
```

## 設定ファイル

実験条件は2種類の JSON で決まる。

- `training_config/*.json`: モデル種別・層数・hidden 次元、データセットのパスと予測対象（`targets`）、入力・出力の範囲（`index`）、損失、ノイズ、隣接拡張などを指定
- `dataset_config/*_meta.json`: 時間刻み `dt`、軌跡長、各特徴量の static / dynamic 区別

### Adjacency Augmentation の例

```json
"adjacency_augmentation": {
  "enabled": true,
  "random_edges_ratio": 0.2,
  "dilation": { "enabled": true, "hop": 2, "start_layer": 5, "head_fraction": 0.5 },
  "global_attention": { "enabled": true, "node_types": ["OBSTACLE"], "sample_ratio": 0.01 }
}
```

注意: トップレベルの `enabled` が `false` の場合、配下の dilation / global_attention の設定はすべて無効になる。

## 実験・評価スクリプト

| スクリプト | 内容 |
|---|---|
| `scripts/train_mp5_mp15_model_grid_sweep.py` | GNN / GT（拡張なし・あり）× 深さ 5・15 × ノード数 2,000–40,000 の一括実験 |
| `scripts/convert_openfoam_to_h5.py` ほか | OpenFOAM / VTU 結果の H5・XDMF 変換 |
| `scripts/evaluate_predictions.py` | 予測結果の評価 |
| `scripts/plot_mesh_resolution_error.py` | メッシュ解像度と RMSE の関係をプロット |
| `scripts/plot_mesh_resolution_inference_time.py` | メッシュ解像度と推論時間の関係をプロット |
| `scripts/calculate_strouhal.py` | 後流の速度変動を FFT し、Strouhal 数を推定 |

## ベース実装と引用

ベース実装: [DonsetPG/graph-physics](https://github.com/DonsetPG/graph-physics)

```bibtex
@misc{garnier2025trainingtransformersmeshbasedsimulations,
  title  = {Training Transformers for Mesh-Based Simulations},
  author = {Paul Garnier and Vincent Lannelongue and Jonathan Viquerat and Elie Hachem},
  year   = {2025},
  eprint = {2508.18051},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url = {https://arxiv.org/abs/2508.18051}
}
```
