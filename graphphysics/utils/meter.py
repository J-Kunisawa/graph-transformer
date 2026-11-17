import numpy as np
import torch


class Meter(object):
    """
    統計計測器（メーター）の基底クラス。
    学習ループや検証プロセスにおいて、各種指標（Loss, RMSE等）をオンライン（逐次的）に
    計測・追跡するための統一インターフェースを定義する抽象クラス。
    """

    def reset(self):
        """メーターの内部統計バッファを初期状態にリセットする。"""
        pass

    def add(self, value):
        """
        新しい計測値をログに記録し、内部統計を更新する。
        
        Args:
            value: ループの現ステップで算出された最新の指標値。
        """
        pass

    def value(self):
        """現在の状態における集計値（平均、標準偏差など）を取得する。"""
        pass


class AverageValueMeter(Meter):
    """
    オンライン平均・標準偏差計測器。
    
    Welfordのオンラインアルゴリズム（Welford's online algorithm）に基づき、
    過去の全入力データをメモリ上に配列として保持することなく、新しく流れ込んでくる値から
    動的に（O(1)のメモリコストで）正確な平均値（Mean）と標本標準偏差（Std）を計算する。
    """
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        """
        最新の指標値（バッチ平均など）をメーターに投入し、走行統計を更新する。

        Args:
            value (float or Tensor): 追加する新しい計測値（またはミニバッチの平均値）。
            n (int): この value に含まれるサンプル数（重み）。デフォルトは 1。
        """
        # PyTorchのテンソル（Tensor）が渡された場合、計算グラフから切り離して純粋な数値（float）に変換
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        
        self.val = value
        
        # サンプル数の累積更新
        old_n = self.n
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
            return

        if old_n == 0:
            # --- データの最初のステップ（初期化） ---
            self.mean = float(value)
            self.m_s = 0.0
            self.std = np.inf if n == 1 else 0.0 # 1データのみの場合は標準偏差は未定義(無限大)
        else:
            # --- Welfordのアルゴリズムによる逐次統計更新（バグ修正版） ---
            # 1. 過去の平均値からの現在の入力値のズレ（偏差）を計算
            delta = value - self.mean
            
            # 2. サンプル数の重み(n)を考慮して、新しい平均値へと移動更新
            self.mean += delta * n / float(self.n)
            
            # 3. 新しい平均値からの偏差を再計算
            delta2 = value - self.mean
            
            # 4. 偏差平方和（M_S: 標本分散の分子となる値）を累積更新
            self.m_s += delta * delta2 * n
            
            # 5. 標本不偏標準偏差（Bessel's correction: 分母を n - 1 にする）を算出
            if self.n > 1:
                self.std = np.sqrt(self.m_s / (self.n - 1.0))
            else:
                self.std = 0.0

    def value(self) -> Tuple[float, float]:
        """現在の累積状態における [平均値, 標準偏差] のタプルを返す。"""
        return self.mean, self.std

    def reset(self):
        """
        全ての統計バッファをクリアし、初期状態（ノード数0、各値 nan）にリセットする。
        各エポックの開始時等に呼び出される。
        """
        self.n = 0          # 処理した総サンプル数
        self.val = 0.0      # 直近に投入された生の入力値
        self.mean = np.nan  # 現在の累積平均値
        self.m_s = 0.0      # 逐次分散計算用の偏差平方和バッファ
        self.std = np.nan   # 現在の累積不偏標準偏差