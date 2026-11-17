import json
import warnings

import torch
from absl import app, flags
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import get_dataset, get_preprocessing

# データ結合時のPyG内部ワーニング（バッチサイズ推定の曖昧さ）を非表示にして推論ログをスッキリさせる
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

# 推論を最速化するためのTensorFloat-32（TF32）設定
torch.set_float32_matmul_precision("high")
# マルチプロセス時のファイルシステム記述子不足を防ぐ安全策
torch.multiprocessing.set_sharing_strategy("file_system")

# --- ABSLライブラリを用いたコマンドライン引数（Flags）の定義 ---
FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "prediction_project", "WandB上に作成する推論プロジェクト名")
flags.DEFINE_string("model_path", None, "読み込む学習済みモデルチェックポイント（.ckpt）のパス（必須）")
flags.DEFINE_bool("no_edge_feature", False, "エッジ特徴量の計算を無効化するかどうか")
flags.DEFINE_string(
    "predict_parameters_path", None, "モデルやインデックス構造が記述された設定JSONファイルのパス（必須）"
)
flags.DEFINE_string(
    "prediction_save_path", "predictions", "生成されたXDMFシミュレーション結果ファイルをエクスポートする保存先フォルダ"
)

# 過去の時刻発展履歴（過去データ）のフィードバック設定
flags.DEFINE_bool("use_previous_data", True, "時系列の発展推論に1ステップ前の履歴（t-1）をフィードバックさせるか")
flags.DEFINE_integer("previous_data_start", 4, "特徴量内における過去データの格納開始インデックス")
flags.DEFINE_integer("previous_data_end", 7, "特徴量内における過去データの格納終了インデックス")
flags.DEFINE_bool("use_test", True, "推論時に test_path を使用するか（True の場合 test_path を強制使用）")


def main(argv):
    del argv # ABSL仕様上の引数処理

    # 必須パラメータファイルのチェック
    if not FLAGS.predict_parameters_path:
        raise ValueError("--predict_parameters_path で設定JSONファイルのパスを指定する必要があります。")
    if not FLAGS.model_path:
        raise ValueError("--model_path で学習済みチェックポイント（.ckpt）のパスを指定する必要があります。")

    # 設定JSONファイルの読み込み
    parameters_path = FLAGS.predict_parameters_path
    logger.info(f"Opening prediction parameters from {parameters_path}")
    try:
        with open(parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"設定JSONファイルの読み込みに失敗しました: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 引数フラグのローカル変数へのアンパック
    wandb_project_name = FLAGS.project_name
    model_path = FLAGS.model_path
    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data
    previous_data_start = FLAGS.previous_data_start
    previous_data_end = FLAGS.previous_data_end
    prediction_save_path = FLAGS.prediction_save_path

    # 外部前処理関数の動的読み込み
    extra_node_features = None
    extra_features_type = parameters.get("transformations", {}).get("preprocessing", {}).get("extra_features")
    if extra_features_type == "aneurysm":
        from graphphysics.external.aneurysm import build_features as extra_node_features
    elif extra_features_type == "panels":
        from graphphysics.external.panels import build_features as extra_node_features
    elif extra_features_type == "bezier":
        from graphphysics.external.bezier import add_bezier_node_type as extra_node_features
    elif extra_features_type == "append_previous":
        from graphphysics.external.append_previous import build_features as extra_node_features

    # --- 1. 推論用前処理関数のビルド（実運用のため、ガウスノイズを付加する処理は完全に排除：remove_noise=True） ---
    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True, # 訓練時と異なり、ノイズを一切混ぜずに綺麗な場を入力
        extra_node_features=extra_node_features, # 幾何特徴量（脳動脈瘤等）の抽出
    )

    # --- 2. 評価用データセットの取得 ---
    # predict: デフォルトで test_path を使用する運用にする（switch_to_val=False）
    use_test = FLAGS.use_test
    predict_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
        switch_to_val=False,
        use_test=use_test,
    )

    # 設定ファイルに記載されている train/valid/test のパスを明示的にログ出力
    dataset_cfg = parameters.get("dataset", {})
    logger.info(f"Configured train_path: {dataset_cfg.get('train_path')}")
    logger.info(f"Configured val_path: {dataset_cfg.get('val_path')}")
    logger.info(f"Configured test_path: {dataset_cfg.get('test_path')}")

    # 実際に読み込まれている dataset のファイルパスをログ出力（h5/xdmf に応じて）
    dataset_path_log = getattr(predict_dataset, "h5_path", None) or getattr(predict_dataset, "xdmf_folder", None)
    logger.info(f"Using dataset path: {dataset_path_log}")

    # --- 3. データローダーの定義 ---
    # 時系列フレームを t=0 から順番に入力して自律回帰（ロールアウト）させるため、
    # shuffle=False（順序固定）、batch_size=1（1つのシミュレーションケースずつ実行）が絶対条件となる。
    predict_dataloader_kwargs = {
        "dataset": predict_dataset,
        "shuffle": False, 
        "batch_size": 1,   
        "num_workers": 10,   # 推論時はファイルの逐次追記I/Oが発生するため、メインプロセス(0)で安全に逐次実行
    }
    predict_dataloader = DataLoader(**predict_dataloader_kwargs)

    # 履歴フィードバック用パラメータのパック
    prev_data_kwargs = {}
    if use_previous_data is True:
        prev_data_kwargs = {
            "use_previous_data": True,
            "previous_data_start": previous_data_start,
            "previous_data_end": previous_data_end,
        }
        
    # --- 4. 学習済みチェックポイントからの完全復元ロード ---
    logger.info(f"Loading model from checkpoint: {model_path}")
    lightning_module = LightningModule.load_from_checkpoint(
        checkpoint_path=model_path,
        parameters=parameters,
        trajectory_length=predict_dataset.trajectory_length,
        timestep=predict_dataset.dt,
        prediction_save_path=prediction_save_path, # ファイルエクスポート先パスを注入
        **prev_data_kwargs,
    )

    # --- 5. Weights & Biases (WandB) ロガーの立ち上げ（推論プロセスの追跡用） ---
    wandb_run = wandb.init(project=wandb_project_name)
    wandb_logger = WandbLogger(experiment=wandb_run)

    # 実験ダッシュボードのコンフィグ情報の上書き同期
    wandb_logger.experiment.config.update(
        {
            "architecture": parameters["model"]["type"],
            "#_layers": parameters["model"]["message_passing_num"],
            "#_neurons": parameters["model"]["hidden_size"],
            "#_hops": parameters["dataset"]["khop"],
        }
    )

    # --- 6. PyTorch Lightning Trainer の起動設定 ---
    # inference_mode=True により、逆伝播用の計算グラフ（Autograd）の構築を完全にシャットアウトし、
    # 完全に推論に特化させることで、VRAM消費量を削減し演算スピードを極限まで高める。
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        devices=1,
        inference_mode=True, 
    )

    # --- 7. 推論実行（内部で自律回帰ループが回り、逐次XDMF/H5ファイルへ予測結果がエクスポートされる） ---
    logger.success("Starting prediction")
    trainer.predict(model=lightning_module, dataloaders=predict_dataloader)


if __name__ == "__main__":
    # マルチプロセス安全性のための spawn 設定
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)