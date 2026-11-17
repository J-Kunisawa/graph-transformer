#!/usr/bin/env bash

# ==============================================================================
#  Cylinder Flow GNN Training Execution Script (2026)
#  - 物理情報駆動(PINN)マルチロス、空間MTP補助学習、VRAM最適化を統合
# ==============================================================================

# 既存のプロセスやエラー発生時にスクリプトを安全に即時停止させる設定
set -e
set -o pipefail

# --- 1. パスと基本環境の設定 ---
PROJECT_ROOT="/workspace" 
CONFIG_PATH="${PROJECT_ROOT}/training_config/xdmf_transformer.json"
LOG_DIR="${PROJECT_ROOT}/logs"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# --- 2. ハードウェア & ライブラリ最適化環境変数の注入 ---
export CUDA_VISIBLE_DEVICES="0" # 使用するGPU IDを指定（複数ある場合は "0,1" など）
export OMP_NUM_THREADS=4        # CPUデータローダーのMKLスレッド競合を防ぎ、オーバーヘッドを削減
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- 3. ハイパーパラメータの定義（Flagsオーバーライド） ---
PROJECT_NAME="cylinder_flow_project"
MODEL_SAVE_NAME="n=10000/re_100-300_transformer_s3_re0.2" # チェックポイント保存時の名前
NUM_EPOCHS=150          # 10ステップから、十分な収束のために30エポックへスケールアップ
BATCH_SIZE=4            # 5-step BPTTのVRAM増加を考慮（1更新あたり20フレーム）
INIT_LR=0.001          # スケジューラのピーク学習率
WARMUP_RATIO=0.02      # 線形ウォームアップ期間の割合
NUM_WORKERS=8          # CPUでの前処理並列プロセス数（Ubuntuワークステーションのコア数に応じて調整）
PREFETCH_FACTOR=2      # CPUがGPUの計算裏で先読みしておくバッチ数
SEED=42                # 乱数シード
MODEL_PATH=""   # 既存チェックポイントから再開する場合のパス
RESUME_TRAINING=false   # 重みのみ引き継ぎ、multi-step用optimizer/early-stopを新規開始
USE_PREVIOUS_DATA=false # 1ステップ前の履歴を使うかどうか
PREVIOUS_DATA_START=3  # 過去データ(差分)の開始インデックス。 [u,v,p, diff_u,diff_v,diff_p, type,...] の diff_u の位置
PREVIOUS_DATA_END=6    # 過去データ(差分)の終了インデックス（Pythonスライスと同様にこのインデックスは含まない）
NO_EDGE_FEATURE=false  # エッジ特徴量を無効化するかどうか
USE_PARTITIONING=false # グラフパーティショニングを使うかどうか
NUM_PARTITIONS=""     # 分割数
MAX_NODES_PER_PARTITION="" # 1部分グラフの最大ノード数
GRADIENT_BATCH_SIZE=1  # 勾配累積ステップ数

# --- 4. 実行ログのタイムスタンプ生成 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXEC_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "🚀 訓練プロセスを開始します"
echo "   - 開始時刻: $(date)"
echo "   - 使用設定: ${CONFIG_PATH}"
echo "   - 出力ログ: ${EXEC_LOG}"
echo "========================================================================"

# --- 5. 訓練コマンドの実行 ---
# `nice -n 10`: システムの全体的なハングを防ぐため、CPUプロセスの優先度をわずかに下げて実行
# `tee`: ログファイルに保存しつつ、標準出力（コンソール）にもリアルタイムで進捗を表示
TRAIN_ARGS=(
    --project_name="${PROJECT_NAME}"
    --training_parameters_path="${CONFIG_PATH}"
    --model_save_name="${MODEL_SAVE_NAME}"
    --resume_training=${RESUME_TRAINING}
    --num_epochs=${NUM_EPOCHS}
    --seed=${SEED}
    --batch_size=${BATCH_SIZE}
    --init_lr=${INIT_LR}
    --warmup_ratio=${WARMUP_RATIO}
    --num_workers=${NUM_WORKERS}
    --prefetch_factor=${PREFETCH_FACTOR}
    --use_previous_data=${USE_PREVIOUS_DATA}
    --previous_data_start=${PREVIOUS_DATA_START}
    --previous_data_end=${PREVIOUS_DATA_END}
    --no_edge_feature=${NO_EDGE_FEATURE}
    --use_partitioning=${USE_PARTITIONING}
    --gradient_batch_size=${GRADIENT_BATCH_SIZE}
)

if [[ -n "${MODEL_PATH}" ]]; then
    TRAIN_ARGS+=(--model_path="${MODEL_PATH}")
fi

if [[ -n "${NUM_PARTITIONS}" ]]; then
    TRAIN_ARGS+=(--num_partitions=${NUM_PARTITIONS})
fi

if [[ -n "${MAX_NODES_PER_PARTITION}" ]]; then
    TRAIN_ARGS+=(--max_nodes_per_partition=${MAX_NODES_PER_PARTITION})
fi

nice -n 10 python3 "${PROJECT_ROOT}/train.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "${EXEC_LOG}"

echo "========================================================================"
echo "🎉 訓練プロセスが正常に終了しました（ログ: ${EXEC_LOG}）"
echo "========================================================================"