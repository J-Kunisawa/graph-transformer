import json
import os
import sys
import warnings
from typing import Optional

import torch
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)
from graphphysics.utils.progressbar import ColabProgressBar

# データ結合時のPyG内部ワーニング（バッチサイズ推定の曖昧さ）を非表示にしてログをクリーンに保つ
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

# A100/H100/RTX等のTensorCoreを活用し、3D行列演算を高速化する設定（TensorFloat-32有効化）
torch.set_float32_matmul_precision("high")
# マルチプロセスDataLoaderでのデータ共有時にファイルシステム枯渇（Too many open files）を防ぐ安全策
torch.multiprocessing.set_sharing_strategy("file_system")

# --- ABSLライブラリを用いたコマンドライン引数（Flags）の定義 ---
FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "WandB上に作成するプロジェクト名")
flags.DEFINE_integer("num_epochs", 10, "訓練を回す総エポック数")
flags.DEFINE_integer("seed", 42, "擬似乱数固定用シード値")
flags.DEFINE_float("init_lr", 0.001, "学習率の初期値")
flags.DEFINE_integer("batch_size", 2, "1ミニバッチあたりのメッシュ数")
flags.DEFINE_integer("warmup", 1000, "学習率スケジューラの線形ウォームアップステップ数")
flags.DEFINE_float("warmup_ratio", -1.0, "学習率スケジューラのウォームアップ期間の割合。0以上で指定すると warmup を上書きする")
flags.DEFINE_integer("num_workers", 2, "データ読み込み・前処理用のCPU並列プロセス数")
flags.DEFINE_integer("prefetch_factor", 2, "CPUワーカーが事前ロードしておくバッチ数の因数")
flags.DEFINE_string("model_save_name", None, "訓練中に保存するチェックポイントのファイル名")
flags.DEFINE_string("model_path", None, "再開、または推論時に読み込むモデルチェックポイント（.ckpt）のパス")
flags.DEFINE_bool("resume_training", False, "中断された訓練をチェックポイントから完全再開するかどうか")

# 過去の時刻発展履歴（過去データ）のフィードバック設定
flags.DEFINE_bool("use_previous_data", True, "時系列の連続性を強化するために1ステップ前の履歴を使用するか")
flags.DEFINE_integer("previous_data_start", 4, "ノード特徴量内における過去データの格納開始インデックス")
flags.DEFINE_integer("previous_data_end", 7, "ノード特徴量内における過去データの格納終了インデックス")
flags.DEFINE_bool("no_edge_feature", False, "エッジ特徴量（相対位置・距離）の計算を無効化するか")
flags.DEFINE_string("training_parameters_path", None, "モデルやロスの詳細設定が記述されたJSONファイルのパス（必須）")

# 大規模メッシュ用のMETIS分割パラメータ
flags.DEFINE_bool("use_partitioning", False, "VRAM節約のためにグラフパーティショニング（分割学習）を行うか")
flags.DEFINE_integer("num_partitions", None, "メッシュ全体の分割数")
flags.DEFINE_integer("max_nodes_per_partition", None, "1つの部分グラフに許容する最大ノード数")
flags.DEFINE_integer("gradient_batch_size", 1, "疑似バッチサイズ拡大のための勾配累積（Gradient Accumulation）ステップ数")


def _resolve_trainer_precision(enable_vram_optimizations: bool) -> Optional[str]:
    """
    GPUのアーキテクチャ性能を検出し、最適な自動混合精度（AMP）のモードを判定する。
    Ampere世代以降であればbf16-mixed、それ以前のGPUであれば16-mixed（FP16）を返す。
    """
    if not enable_vram_optimizations or not torch.cuda.is_available():
        return None
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    return "bf16-mixed" if is_bf16_supported() else "16-mixed"


def main(argv):
    del argv # ABSL仕様上の引数処理

    # 必須パラメータファイルのチェック
    if not FLAGS.training_parameters_path:
        raise ValueError("--training_parameters_path でJSON設定ファイルのパスを指定する必要があります。")

    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"設定JSONファイルの読み込みに失敗しました: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 引数フラグのローカル変数へのアンパック
    wandb_project_name = FLAGS.project_name
    num_epochs = FLAGS.num_epochs
    initial_lr = FLAGS.init_lr
    batch_size = FLAGS.batch_size
    warmup = FLAGS.warmup
    warmup_ratio = FLAGS.warmup_ratio
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_name = FLAGS.model_save_name
    model_path = FLAGS.model_path
    resume_training = FLAGS.resume_training
    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data
    previous_data_start = FLAGS.previous_data_start
    previous_data_end = FLAGS.previous_data_end
    use_partitioning = FLAGS.use_partitioning
    num_partitions = FLAGS.num_partitions
    max_nodes_per_partition = FLAGS.max_nodes_per_partition
    gradient_batch_size = FLAGS.gradient_batch_size

    # JSON内のグローバル訓練オプションと引数フラグの同期
    training_params = parameters.setdefault("training", {})
    enable_vram_optimizations = bool(training_params.get("enable_vram_optimizations", False))
    training_params["enable_vram_optimizations"] = enable_vram_optimizations
    early_stopping_params = training_params.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_params.get("enabled", False))
    early_stopping_patience = int(early_stopping_params.get("patience", 20))
    early_stopping_min_delta = float(early_stopping_params.get("min_delta", 0.0))
    multi_step_enabled = bool(training_params.get("multi_step_loss", False))
    multi_step_steps = int(training_params.get("multi_step_loss_steps", 5))
    if multi_step_enabled and multi_step_steps < 2:
        raise ValueError("training.multi_step_loss_steps は2以上である必要があります。")
    if early_stopping_patience < 1:
        raise ValueError("training.early_stopping.patience は1以上である必要があります。")
    if early_stopping_min_delta < 0.0:
        raise ValueError("training.early_stopping.min_delta は0以上である必要があります。")

    # 乱数シードの完全固定（PyGのデータシャッフルやモデル初期化の再現性を確保）
    seed_everything(FLAGS.seed, workers=True)

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

    # --- 1. 訓練用データセットの構築（ノイズ拡張ON） ---
    train_preprocessing = get_preprocessing(
        param=parameters, 
        device=device, 
        use_edge_feature=use_edge_feature,
        extra_node_features=extra_node_features, 
    )
    train_dataset = get_dataset(
        param=parameters, 
        preprocessing=train_preprocessing, 
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data, 
        use_partitioning=use_partitioning,
        num_partitions=num_partitions, 
        max_nodes_per_partition=max_nodes_per_partition,
        multi_step_steps=multi_step_steps if multi_step_enabled else 1,
    )

    # --- 2. 検証用データセットの構築（時間の連続評価を壊さないようノイズ付加はOFF） ---
    val_preprocessing = get_preprocessing(
        param=parameters, 
        device=device, 
        use_edge_feature=use_edge_feature,
        remove_noise=True, # ノイズの除去
        extra_node_features=extra_node_features,
    )
    val_dataset = get_dataset(
        param=parameters, preprocessing=val_preprocessing, use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data, switch_to_val=True, # 検証データ用パスへ切り替え
    )

    num_workers = get_num_workers(param=parameters, default_num_workers=num_workers)

    # PyG専用のDataLoaderキーワード引数の設定
    train_dataloader_kwargs = {
        "dataset": train_dataset,
        "shuffle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "exclude_keys": ["tetra"], # ミニバッチ結合エラーを防ぐため、3D四面体セル生情報は除外
    }

    valid_dataloader_kwargs = {
        "dataset": val_dataset,
        "shuffle": False,
        "batch_size": 1, # 自律回帰（ロールアウト）を1ケースずつ評価するためバッチサイズは1で固定
        "num_workers": num_workers,
    }

    # HDF5形式データセットの場合、GPUへの転送を高速化するためにピンメモリ（非同期メモリプール）を有効化
    if train_dataset.type == "h5":
        pin_memory = device.type == "cuda"
        if os.getenv("FORCE_PIN_MEMORY", "1") == "0":
            pin_memory = False
        train_dataloader_kwargs["pin_memory"] = pin_memory
        valid_dataloader_kwargs["pin_memory"] = pin_memory

    # マルチワーカー時のプリフェッチ（先読み並列パイプライン）の適用
    if num_workers > 0:
        train_dataloader_kwargs.update({"prefetch_factor": prefetch_factor, "persistent_workers": True})
        valid_dataloader_kwargs.update({"prefetch_factor": prefetch_factor, "persistent_workers": True})

    # データローダーの生成
    train_dataloader = DataLoader(**train_dataloader_kwargs)
    valid_dataloader = DataLoader(**valid_dataloader_kwargs)

    # 最適化スケジューラに引き渡す総訓練ステップ数の厳密な事前計算
    num_steps = num_epochs * len(train_dataloader) // gradient_batch_size

    prev_data_kwargs = {}
    if use_previous_data is True:
        prev_data_kwargs = {
            "use_previous_data": True,
            "previous_data_start": previous_data_start,
            "previous_data_end": previous_data_end,
        }

    # --- 3. システムモジュール（LightningModule）のロードまたは新規立ち上げ ---
    if model_path and os.path.isfile(model_path):
        logger.info(f"既存のチェックポイントから状態を復旧します: {model_path}")
        load_kwargs = dict(
            checkpoint_path=model_path,
            parameters=parameters,
            warmup=warmup,
            warmup_ratio=warmup_ratio,
            learning_rate=initial_lr,
            num_steps=num_steps,
            trajectory_length=train_dataset.trajectory_length,
            timestep=train_dataset.dt,
            **prev_data_kwargs,
        )
        try:
            lightning_module = LightningModule.load_from_checkpoint(
                strict=True,
                **load_kwargs,
            )
        except RuntimeError as exc:
            logger.warning(
                "チェックポイントの state_dict が現在のモデル定義と完全一致しません。 "
                "互換性を優先して strict=False で再読み込みします。"
            )
            logger.warning(f"strict=True の失敗理由: {exc}")
            lightning_module = LightningModule.load_from_checkpoint(
                strict=False,
                **load_kwargs,
            )
        logger.info(f"WandBの過去ログ系列を復元再接続します。Run ID: {lightning_module.wandb_run_id}")
    else:
        logger.info("新規に物理AIモデルを初期化します")
        lightning_module = LightningModule(
            parameters=parameters, 
            learning_rate=initial_lr, 
            num_steps=num_steps, 
            warmup=warmup,
            warmup_ratio=warmup_ratio,
            trajectory_length=train_dataset.trajectory_length, 
            timestep=train_dataset.dt,
            **prev_data_kwargs,
        )

    # --- 4. Weights & Biases (WandB) 実験管理システムの初期化 ---
    if resume_training:
        # 中断再開時は、過去のチェックポイントからサルベージした一意の run_id でセッションを再開
        wandb_run = wandb.init(
            project=wandb_project_name, id=lightning_module.wandb_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=wandb_project_name)

    wandb_logger = WandbLogger(experiment=wandb_run)
    lightning_module.wandb_run_id = wandb_logger.experiment.id # 新規セッションIDをモジュールへ記録
    
    # モデルの自動チェックポイント保存Callbackの設定
    # 検証時の総ロールアウトRMSE（val_all_rollout_rmse）が最小のときのモデルを保存するように設定
    monitor_metric = "val_all_rollout_rmse"
    if model_save_name is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename=model_save_name,
            monitor=monitor_metric,
            mode="min",
            save_top_k=1,
            save_last=False,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            monitor=monitor_metric,
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer_callbacks = [
        ColabProgressBar(), # Colab環境用クリーンバー
        checkpoint_callback,
        lr_monitor,
    ]
    if early_stopping_enabled:
        trainer_callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                mode="min",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                strict=True,
                verbose=True,
            )
        )

    # 実験メタ情報のWandB構成ページへの自動パッキング
    wandb_logger.experiment.config.update(
        {
            "architecture": parameters["model"]["type"],
            "#_layers": parameters["model"]["message_passing_num"],
            "#_neurons": parameters["model"]["hidden_size"],
            "#_hops": parameters["dataset"]["khop"],
            "max_lr": initial_lr,
            "batch_size": batch_size,
            "enable_vram_optimizations": enable_vram_optimizations,
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "multi_step_loss": multi_step_enabled,
            "multi_step_loss_steps": multi_step_steps if multi_step_enabled else 1,
        },
        # 再開時は、チェックポイント作成後に変更した実行設定（例: VRAM最適化）を
        # 既存runへ反映できるようにする。新規runでは従来どおり変更を禁止する。
        allow_val_change=resume_training,
    )

    # 混合精度の解決結果を反映
    trainer_precision = _resolve_trainer_precision(enable_vram_optimizations)
    if enable_vram_optimizations and trainer_precision is None:
        logger.warning("省メモリ最適化が要求されましたが、CUDA(GPU)環境が無いため混合精度を無効化します。")

    # --- 5. PyTorch Lightning Trainer の最終構成組み上げ ---
    trainer_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "max_epochs": num_epochs,
        "logger": wandb_logger,
        "callbacks": trainer_callbacks,
        "log_every_n_steps": 100,
        "gradient_clip_val": 1.0, # 長期シミュレーションでの勾配爆発を防ぐためのグラジエントクリッピング
        "accumulate_grad_batches": gradient_batch_size, # 勾配累積の適用
    }
    if trainer_precision is not None:
        trainer_kwargs["precision"] = trainer_precision
        wandb_logger.experiment.config.update(
            {"trainer_precision": trainer_precision},
            allow_val_change=resume_training,
        )

    trainer = Trainer(**trainer_kwargs)

    # --- 6. 訓練の実行トリガー（レジューム判定に伴うルート分岐） ---
    if model_path and os.path.isfile(model_path) and resume_training:
        logger.success("チェックポイントの状態（エポック、オプティマイザ）を引き継いで訓練を再開します。")
        trainer.fit(
            model=lightning_module, 
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader, 
            ckpt_path=model_path,
        )
    else:
        logger.success("新規のタイムスタンプから訓練セッションを開始します。")
        trainer.fit(
            model=lightning_module, 
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


if __name__ == "__main__":
    # マルチプロセス DataLoader が CUDA テンソルを安全にやり取りできるようにプロセス開始方式を 'spawn' に強制
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
