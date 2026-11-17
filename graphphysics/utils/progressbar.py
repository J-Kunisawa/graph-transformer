from lightning.pytorch.callbacks import TQDMProgressBar


class ColabProgressBar(TQDMProgressBar):
    """
    Google Colaboratory / Jupyter Notebook 向けのカスタム・プログレスバー。
    
    PyTorch Lightning 標準の TQDMProgressBar を拡張し、
    検証・推論・テスト時にブラウザの出力ログが何行も生成されて肥大化（スクロール崩壊やフリーズ）
    するのを防ぐため、それらの進捗バーを「非表示（disable）」にする。
    学習バーの指標は最新ステップの学習損失と直近の検証平均損失に絞り、行の折り返しを抑える。
    """

    def get_metrics(self, trainer, pl_module):
        # 表示だけを絞り込む。WandBへの指標名・記録内容は変更しない。
        metrics = super().get_metrics(trainer, pl_module)
        displayed_metrics = {
            "loss": value
            for name, value in metrics.items()
            if name.startswith("train_") and name.endswith("_step")
        }
        if "val_loss_epoch" in metrics:
            displayed_metrics["val_loss"] = metrics["val_loss_epoch"]
        return displayed_metrics

    def init_validation_tqdm(self):
        """
        検証（Validation）ループ実行時の tqdm プログレスバーを初期化・設定するメソッド。
        親クラスのバーを立ち上げた直後に disable=True にして出力をシャットアウトする。
        """
        bar = super().init_validation_tqdm()
        bar.disable = True  # 検証ステップごとの進捗バー出力を完全に無効化
        return bar

    def init_predict_tqdm(self):
        """
        推論（Predict / 自律回帰ロールアウトによるファイル出力）ステップ実行時の
        tqdm プログレスバーを初期化するメソッド。同様に進捗バーの生成を無効化する。
        """
        bar = super().init_predict_tqdm()
        bar.disable = True  # 推論実行時の進捗バー出力を完全に無効化
        return bar

    def init_test_tqdm(self):
        """
        テスト（Test / 評価用データセットの最終検証）ステップ実行時の
        tqdm プログレスバーを初期化するメソッド。
        """
        bar = super().init_test_tqdm()
        bar.disable = True  # テスト実行時の進捗バー出力を完全に無効化
        return bar