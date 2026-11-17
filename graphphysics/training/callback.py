import os
from typing import List

import lightning.pytorch as pl
import numpy as np
import pyvista as pv
import torch
from lightning.pytorch.callbacks import Callback
from torch_geometric.data import Data, Dataset

import wandb
from graphphysics.training.lightning_module import build_mask
from graphphysics.utils.pyvista_mesh import convert_to_pyvista_mesh


class LogPyVistaPredictionsCallback(Callback):
    """
    PyTorch Lightning コールバッククラス。
    各検証（Validation）エポックの終了時に、モデルの予測した物理場（速度場コンター等）を
    PyVistaによる3Dレンダリングを用いて画像および動画化し、Weights & Biases (WandB) へ自動送信する。
    """

    def __init__(
        self, dataset: Dataset, indices: List[int], output_dir: str = "predictions"
    ):
        """
        パラメータ:
            dataset (Dataset): 検証データを取得するためのPyGデータセット。
            indices (List[int]): 静止画ログ、および動画化するフレーム範囲を指定するためのインデックスリスト。
            output_dir (str): 生成された画像等の一時保存先ディレクトリ。デフォルトは 'predictions'。
        """
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.output_dir = output_dir

    def _forward_with_autocast_if_needed(
        self, model: pl.LightningModule, graph: Data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        モデルの重みが半精度（Float16 / BFloat16）で訓練されている場合、
        コールバック側の前方向計算でも自動混合精度（CUDA Autocast）のコンテキストを安全に展開して推論を行う。
        """
        first_param = next(model.parameters(), None)
        use_cuda_autocast = (
            graph.x.is_cuda
            and first_param is not None
            and first_param.dtype in (torch.float16, torch.bfloat16)
        )
        if use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=first_param.dtype):
                return model(graph)
        return model(graph)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        検証エポックの終了時にPyTorch Lightningシステムから自動的に呼び出されるメインループ。
        """
        os.makedirs(self.output_dir, exist_ok=True)
        model = pl_module
        device = model.device

        images = []
        ground_truth = []

        # --- フェーズ 1: 指定された特定サンプルの「静止画像」の生成・ログ ---
        with torch.no_grad():
            for idx in self.indices:
                # 指定インデックスの生グラフを抽出し、GPUなどのデバイスへ転送
                graph = self.dataset[idx].to(device)
                
                # シミュレータに入力して予測を実行（内部で正規化や変化量の加算が処理される）
                _, _, predicted_outputs = self._forward_with_autocast_if_needed(
                    model, graph
                )

                # グラフのノード特徴量を、モデルが算出した「実スケールの予測物理量」に上書き
                graph.x = predicted_outputs
                
                feature_names = model.param.get("dataset", {}).get("feature_names", None)

                # PyVista形式の3Dメッシュデータオブジェクトへ変換
                predicted_mesh = self._convert_to_pyvista_mesh(graph, feature_names)

                # オフスクリーン（画面非表示）で3Dレンダリングを行い、スクリーンショット画像を取得
                img = self._generate_pyvista_image(predicted_mesh, feature_names)
                images.append(
                    wandb.Image(img, caption=f"Prediction: Sample Index: {idx}")
                )

                # --- 全く同じ処理を正解データ（Ground Truth）に対しても実行 ---
                graph.x = graph.y # 特徴量を次の時刻の正解物理量に書き換え
                ground_truth_mesh = self._convert_to_pyvista_mesh(graph, feature_names)

                img = self._generate_pyvista_image(ground_truth_mesh, feature_names)
                ground_truth.append(
                    wandb.Image(img, caption=f"Ground Truth: Sample Index: {idx}"),
                )

        # WandBのロガーを通じて、予測と正解の画像を比較可能な形でダッシュボードへ送信
        wandb_logger = trainer.logger
        wandb_logger.experiment.log({"pyvista_predictions": images})
        wandb_logger.experiment.log({"pyvista_ground_truth": ground_truth})

        frames_predictions = []
        frames_ground_truth = []
        predicted_outputs = None

        # --- フェーズ 2: 連続フレームを用いた「時系列ロールアウト動画（シミュレーションアニメーション）」の生成 ---
        if len(self.indices) > 1:
            with torch.no_grad():
                # 指定された最初のインデックスから最後のインデックスまでを時間軸としてループ
                for idx in range(self.indices[0], self.indices[-1]):
                    graph = self.dataset[idx].to(device)

                    # 2ステップ目以降：前ステップでAIが予測した値を、次の入力特徴量フィールドへ動的に書き戻す（自律回帰の接続）
                    if predicted_outputs is not None:
                        graph.x[
                            :,
                            model.model.output_index_start : model.model.output_index_end,
                        ] = predicted_outputs.detach()

                    # 壁面や流入条件などの不変境界のインデックスを特定するためのマスクを作成
                    mask = build_mask(model.param, graph)
                    target = graph.y # 次のステップの正解物理量

                    # 1ステップ先を予測
                    _, _, predicted_outputs = self._forward_with_autocast_if_needed(
                        model, graph
                    )
                    
                    # 誤差累積の防波堤：壁面等の境界条件ノードの予測値を、正解の値（流速0など）に強制リセット・固定
                    predicted_outputs[mask] = target[mask]

                    graph.x = predicted_outputs
                    
                    feature_names = model.param.get("dataset", {}).get("feature_names", None)

                    # 予測メッシュの3D変換とレンダリング画像配列の取得
                    predicted_mesh = self._convert_to_pyvista_mesh(graph, feature_names)
                    img = self._generate_pyvista_image(predicted_mesh, feature_names)
                    
                    # 動画配列表信用に NumPy型 (uint8: 0~255) に変換してリストへストック
                    img_array = np.array(img).astype(np.uint8)
                    frames_predictions.append(img_array)

                    # --- 正解データ（Ground Truth）側のアニメーションフレームも同様に作成 ---
                    graph.x = graph.y
                    ground_truth_mesh = self._convert_to_pyvista_mesh(graph, feature_names)

                    img = self._generate_pyvista_image(ground_truth_mesh, feature_names)
                    img_array = np.array(img).astype(np.uint8)
                    frames_ground_truth.append(img_array)

            # 収集した全フレームの画像を時間軸方向にスタックして4次元テンソルを形成
            # 形状: (時間ステップ数, 高さ, 幅, チャンネルRGB)
            frames_predictions = np.stack(frames_predictions, axis=0)  
            frames_ground_truth = np.stack(frames_ground_truth, axis=0)

            # WandB Videoオブジェクトが要求するテンソル規格 (時間, チャンネル, 高さ, 幅) へ軸を入れ替え
            frames_predictions = np.transpose(frames_predictions, (0, 3, 1, 2))
            frames_ground_truth = np.transpose(frames_ground_truth, (0, 3, 1, 2))

            # 秒間4フレーム (fps=4) の流体アニメーション動画オブジェクトを構築
            video_predictions = wandb.Video(frames_predictions, fps=4)
            video_ground_truth = wandb.Video(frames_ground_truth, fps=4)

            # WandBへ予測シミュレーション動画と正解シミュレーション動画をアップロード
            wandb_logger = trainer.logger
            wandb_logger.experiment.log(
                {"pyvista_predictions_video": video_predictions}
            )
            wandb_logger.experiment.log(
                {"pyvista_ground_truth_video": video_ground_truth}
            )

    def _convert_to_pyvista_mesh(self, graph: Data, feature_names: List[str] = None) -> pv.PolyData:
        """
        PyTorch Geometric の Data 構造（テンソル）から、PyVistaの 3D多角形メッシュデータ構造（PolyData）へ変換する。
        """
        # 外部のユーティリティ関数を使って基本幾何トポロジーを変換
        mesh = convert_to_pyvista_mesh(graph=graph, add_all_data=True, feature_names=feature_names)

        return mesh

    def _generate_pyvista_image(self, predicted_mesh: pv.PolyData, feature_names: List[str] = None):
        """
        PyVistaのプロッターを起動し、メッシュ上に割り当てられた物理量のスカラー場（"x0"）を
        カラーコンター図（色付き分布図）としてレンダリングし、画像配列を抽出して返す。
        """
        # クラスタやサーバー上（画面を持たない環境）でもレンダリングできるよう off_screen=True を設定
        plotter = pv.Plotter(off_screen=True)
        
        # 描画対象の変数を決定（Velocity_Magnitudeが存在すれば優先、無ければ最初の特徴量）
        target_scalar = "Velocity_Magnitude" if "Velocity_Magnitude" in predicted_mesh.point_data else (feature_names[0] if feature_names else "x0")

        # 3D空間に色付きメッシュを追加。不透明度を0.8に設定してわずかに透過させる
        plotter.add_mesh(predicted_mesh, scalars=target_scalar, label=target_scalar, opacity=0.8)
        plotter.add_legend() # カラーバーや凡例の配置
        
        # レンダリング結果をスクリーンショット（画像配列）として取得
        img = plotter.screenshot(return_img=True)
        plotter.close() # メモリリークを防ぐため、プロッターのコンテキストを確実にシャットダウン
        return img