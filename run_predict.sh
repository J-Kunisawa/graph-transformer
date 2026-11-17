#!/usr/bin/env bash

# ==============================================================================
#  Cylinder Flow GNN Inference & Rollout Export Script (2026)
#  - 学習済みモデルを用いた完全自律回帰（時系列予測）と結果のエクスポート
# ==============================================================================

# エラー発生時にスクリプトを安全に即時停止させる設定
set -e
set -o pipefail

# --- 1. パスと基本環境の設定 ---
PROJECT_ROOT="/workspace" # 実際の環境の絶対パスに合わせて適宜変更してください
CONFIG_PATH="${PROJECT_ROOT}/training_config/xdmf_transformer.json" # 推論パラメータは訓練と同じものを使用
LOG_DIR="${PROJECT_ROOT}/logs"
PREDICT_BASE_DIR="${PROJECT_ROOT}/predictions/N=5000/re_80-400/re_100-300_transformer_s3" # 推論結果の保存先ディレクトリ

mkdir -p "${LOG_DIR}"

# --- 2. ハードウェア設定 ---
export CUDA_VISIBLE_DEVICES="0" # 推論に使用するGPU IDを指定
export OMP_NUM_THREADS=1        # 推論時はバッチサイズ1の逐次I/Oになるため、スレッド数は1で最適化

# --- 3. ハイパーパラメータとチェックポイントの定義 ---
PROJECT_NAME="prediction_project"
# ⚠️ 訓練によって生成された最新、または最良のチェックポイントファイルheckpoのパスを指定してください
MODEL_CHECKPOINT="${PROJECT_ROOT}/checkpoints/n=5000/re_100-300_transformer_m1.ckpt"

# --- 4. 実行前チェック ---
if [ ! -f "${MODEL_CHECKPOINT}" ]; then
    echo "❌ エラー: 指定されたモデルチェックポイントが見つかりません:"
    echo "   ${MODEL_CHECKPOINT}"
    echo "   まずは学習を完了させるか、正しいパスを指定してください。"
    exit 1
fi

# --- 5. 実行ログのタイムスタンプ生成 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXEC_LOG="${LOG_DIR}/predict_${TIMESTAMP}.log"
PREDICT_OUT_DIR="${PREDICT_BASE_DIR}/${TIMESTAMP}"
mkdir -p "${PREDICT_OUT_DIR}"

echo "========================================================================"
echo "🔮 推論プロセスを開始します"
echo "   - 開始時刻: $(date)"
echo "   - 読込モデル: ${MODEL_CHECKPOINT}"
echo "   - 出力先フォルダ: ${PREDICT_OUT_DIR}"
echo "   - 出力ログ: ${EXEC_LOG}"
echo "========================================================================"

# --- 6. 推論コマンドの実行 ---
# `inference_mode=True` で内部駆動するため、VRAMや計算グラフの無駄を排除して超高速に処理されます
python3 "${PROJECT_ROOT}/predict.py" \
    --project_name="${PROJECT_NAME}" \
    --predict_parameters_path="${CONFIG_PATH}" \
    --model_path="${MODEL_CHECKPOINT}" \
    --prediction_save_path="${PREDICT_OUT_DIR}" \
    --use_previous_data=false \
    --previous_data_start=4 \
    --previous_data_end=7 \
    --no_edge_feature=false \
    2>&1 | tee "${EXEC_LOG}"

echo "========================================================================"
echo "🎉 推論およびシミュレーションファイルのエクスポートが完了しました（ログ: ${EXEC_LOG}）"
echo "========================================================================"