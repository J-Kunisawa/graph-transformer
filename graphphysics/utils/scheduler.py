from typing import List

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    線形ウォームアップ付きコサインアニーリング学習率スケジューラ。
    
    1. 訓練初期（warmupステップ）：学習率を 0 から最大学習率（1.0倍）まで直線的に増加させ、
       初期の巨大な勾配による計算のバースト（発散）を防ぐ。
    2. 訓練中期〜終期：コサイン関数に沿って、最大学習率から下限（min_lr_factor）まで
       なめらかに学習率を減衰させ、局所解を回避しつつ精密な収束を図る。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        max_iters: int,
        min_lr_factor: float = 0.001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): ラップする最適化関数（AdamW等）。
            warmup (int): 初期ウォームアップを実行する総反復（ステップ）数。
            max_iters (int): 訓練全体の総反復（ステップ）数。
            min_lr_factor (float): 到達する最小学習率の係数（base_lrに対する比率）。デフォルトは 0.001。
            last_epoch (int): 最後のステップインデックス。初期値は -1。
        """
        self.warmup = warmup
        self.max_iters = max_iters
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """現在のステップに対応する、各パラメータグループ毎の更新後学習率のリストを算出する。"""
        # 現在の進捗ステップ（self.last_epoch）からスケールファクターを計算
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        現在のステップ（epoch）に適合する学習率のスケール因数（0.0 〜 1.0）を
        ウォームアップとコサイン減衰の二段階フェーズで精密に算定する（バグ修正版）。
        """
        epoch += 1  # PyTorch内部仕様として last_epoch が -1 から開始するため、1ベースに補正

        # --- フェーズ 1: 線形ウォームアップ期間 (0 -> 1.0 まで一直線に上昇) ---
        if epoch <= self.warmup:
            if self.warmup == 0:
                return 1.0
            return float(epoch) / float(self.warmup)

        # --- フェーズ 2: コサインアニーリング期間 (1.0 -> min_lr_factor まで滑らかに降下) ---
        # ウォームアップ終了から全体の最大ステップまでの「残りの進捗度」を 0.0 〜 1.0 で規格化
        if self.max_iters <= self.warmup:
            progress = 1.0
        else:
            progress = float(epoch - self.warmup) / float(self.max_iters - self.warmup)
        # 1.0から進行度に応じて徐々に下がるコサインカーブの数式（下限クリップも考慮）
        progress = min(progress, 1.0) # 総ステップ超過時のセーフティ
        
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        # 最終的な因数を計算（最大学習率から最小学習率までの幅にコサインの減衰率を掛ける）
        lr_factor = self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine_decay
        
        return lr_factor