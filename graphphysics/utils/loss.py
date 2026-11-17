import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.vectorial_operators import (
    compute_divergence,
    compute_gradient,
    compute_vector_gradient_product,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_mask_for_loss(
    network_output: torch.Tensor,
    node_type: torch.Tensor,
    masks: list[NodeType],
    selected_indexes: torch.Tensor = None,
):
    """
    損失関数の計算対象となるノードを指定するバイナリマスクを作成するヘルパー関数。
    - masks に指定された属性（例：NORMAL, OUTFLOW）に一致するノードを抽出。
    - selected_indexes が指定されている場合（MeshMask等で隠蔽された領域など）、そのノードを計算から除外する。
    """
    if not masks:
        raise ValueError("損失計算対象のNodeTypeを1つ以上指定してください。")
    node_type = node_type.reshape(-1)
    mask = torch.zeros_like(node_type, dtype=torch.bool)
    for node_type_value in masks:
        mask = torch.logical_or(mask, node_type == int(node_type_value))

    # 特定ノードの除外インデックスが指定されている場合
    if selected_indexes is not None:
        n, _ = network_output.shape
        # selected_indexes に含まれない（~）ノードを示すマスクを作成して結合(AND)
        selected_indexes = torch.as_tensor(
            selected_indexes, dtype=torch.long, device=node_type.device
        )
        nodes_mask = ~torch.isin(
            torch.arange(n, device=node_type.device), selected_indexes
        )
        mask = torch.logical_and(nodes_mask, mask)

    return mask


def _safe_masked_mean(errors: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """空マスクでもNaNを作らず、勾配接続を保ったゼロ損失を返す。"""
    if errors.numel() == 0:
        return reference.sum() * 0.0
    return errors.mean()


class L2Loss(_Loss):
    """標準的な平均二乗誤差（MSE）。予測物理量と正解物理量の全ノードにわたるL2距離を計算する。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # 計算対象ノードの切り出し
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        # 指定属性のノードのみのMSE平均値を算出
        errors = ((network_output - target) ** 2)[mask]
        return _safe_masked_mean(errors, network_output)


class CosineLoss(_Loss):
    """コサイン類似度損失。ベクトルとしての向き（流れの方向）の正確さを最適化する損失関数。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cos = nn.CosineEmbeddingLoss(reduction="none")

    @property
    def __name__(self):
        return "Cosine"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        # 二つのベクトルが「同じ向き（1）」を向くことを目指してEmbeddingLossを解く
        target_tensor = torch.ones(
            target.shape[0], device=mask.device, dtype=target.dtype
        )
        errors = self.cos(network_output, target, target_tensor)[mask]
        return _safe_masked_mean(errors, network_output)


class L1SmoothLoss(_Loss):
    """Smooth L1 損失（Huber損失）。誤差が小さい時はL2、大きい時はL1として振る舞い、外れ値に対して堅牢。"""
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    @property
    def __name__(self):
        return "L1Smooth"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = F.smooth_l1_loss(
            network_output, target, reduction="none", beta=self.beta
        )[mask]
        return _safe_masked_mean(errors, network_output)


class GradientL2Loss(_Loss):
    """
    【物理情報損失：空間勾配】
    AIが予測した実物理場の空間勾配（変化率：∇u）と、正解の空間勾配との間のL2誤差。
    物理場が不自然にギザギザしたり急激に跳ね上がったりするのを抑制し、滑らかな場（Smoothness）を強制する。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "GradientL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )

        # 頂点上の差分演算子を用いて、メッシュ上での空間勾配（Gradient）を計算
        if network_output_gradient is None:
            network_output_gradient = compute_gradient(
                graph, network_output_physical, method=gradient_method, device=device
            )
        if target_gradient is None:
            target_gradient = compute_gradient(
                graph, target_physical, method=gradient_method, device=device
            )
            
        # 勾配同士の差の2乗和平均を計算
        errors = ((network_output_gradient - target_gradient) ** 2)[mask]
        return _safe_masked_mean(errors, network_output_physical)


class ConvectionL2Loss(_Loss):
    """
    【物理情報損失：流体対流項】
    ナビエ・ストークス方程式の「対流項 (u・∇)u」における、予測場と正解場の整合性を保証するL2損失。
    流体の自律的な流動運動（非線形な涡の輸送現象など）の物理的整合性を担保する。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "ConvectionL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        
        # 速度ベクトル場と、その勾配マトリクスから、対流ベクトル項 (u・∇)u を動的に内積演算
        network_output_convection = compute_vector_gradient_product(
            graph, network_output_physical, gradient=network_output_gradient,
            method=gradient_method, device=device,
        )
        target_convection = compute_vector_gradient_product(
            graph, target_physical, gradient=target_gradient,
            method=gradient_method, device=device,
        )
        
        errors = ((network_output_convection - target_convection) ** 2)[mask]
        return _safe_masked_mean(errors, network_output_physical)


class DivergenceL2Loss(_Loss):
    """
    【物理情報損失：質量保存則 (L2規格)】
    AIが予測した速度場の「発散 (Divergence：∇・u)」の2乗和誤差。
    水などの非圧縮性流体では ∇・u = 0 (質量保存則・連続の式) となる必要があるため、
    この出力を0へ向かってペナルティを課すことで、湧き出しのない物理的に正しい流速場を強制する。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "DivergenceL2Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        # メッシュ上での速度場の発散（スカラーフィールド）を計算
        divergence = compute_divergence(
            graph, network_output_physical, gradient=network_output_gradient,
            method=gradient_method, device=device,
        )
        # ∇・u = 0 を目指すため、予測値の2乗そのものをペナルティとする
        errors = (divergence**2)[mask]
        return _safe_masked_mean(errors, network_output_physical)


class DivergenceL1Loss(_Loss):
    """
    【物理情報損失：質量保存則 (L1規格)】
    速度場の発散(∇・u)の絶対値（L1ノルム）損失。L2規格に比べて、局所的な強い湧き出し（外れ値的なスパイク）
    をより厳格に厳しく抑え込みたい場合に有効な、連続の式の制約関数。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "DivergenceL1Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph, network_output_physical, gradient=network_output_gradient,
            method=gradient_method, device=device,
        )
        # 絶対値による一様ペナルティ
        errors = torch.abs(divergence)[mask]
        return _safe_masked_mean(errors, network_output_physical)


class DivergenceL1SmoothLoss(_Loss):
    """【物理情報損失：質量保存則 (Smooth L1規格)】発散の微小なブレは許容しつつ、大きな破綻を滑らかに制御する制約関数。"""
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    @property
    def __name__(self):
        return "DivergenceL1Smooth"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph, network_output_physical, gradient=network_output_gradient,
            method=gradient_method, device=device,
        )
        zeros = torch.zeros_like(divergence)
        # ゼロベクトル場との Huber 誤差を算出
        errors = F.smooth_l1_loss(divergence, zeros, reduction="none", beta=self.beta)[
            mask
        ]
        return _safe_masked_mean(errors, network_output_physical)


class MultiLoss(_Loss):
    """
    【物理情報駆動：PINN統合ブレンドロス関数】
    
    設定JSONに従い、複数の損失関数（例：L2データ誤差 ＋ Divergence質量保存ロス ＋ 空間滑らかさロス）を
    それぞれの固有のブレンド重み（weights）を掛け合わせながら1つに合算（複合最適化）する。
    重複する高コストな幾何勾配計算（compute_gradient）の事前キャッシュ層としても機能する。
    """
    def __init__(self, losses, weights, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        self.weights = weights

    @property
    def __name__(self):
        return "MultiLoss"

    def forward(
        self,
        graph: Data = None,
        network_output_physical: torch.Tensor = None,
        target_physical: torch.Tensor = None,
        gradient_method: str = None,
        return_all_losses: bool = False,
        **kwargs
    ) -> torch.Tensor:

        network_output_gradient = None
        target_gradient = None

        # 性能向上のための事前キャッシュ戦略：
        # 各小損失クラス（GradientLoss, ConvectionLoss等）が内部で個別に compute_gradient を回すと
        # 計算効率が非常に悪くなるため、親であるこのクラスのトップ層で1度だけ空間微分行列を計算し、下に分配する。
        if gradient_method is not None:
            network_output_gradient = compute_gradient(
                graph=graph, field=network_output_physical, method=gradient_method, device=device,
            )
            target_gradient = compute_gradient(
                graph=graph, field=target_physical, method=gradient_method, device=device,
            )

        # 事前計算した勾配をキーワード引数（network_output_gradient=...）を介して各子ロスに流し込み、重み付きで計算
        losses = [
            w
            * loss(
                graph=graph,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=gradient_method,
                network_output_gradient=network_output_gradient,
                target_gradient=target_gradient,
                **kwargs
            )
            for w, loss in zip(self.weights, self.losses)
        ]
        
        # 全てのロスの総和を算出
        errors = sum(losses)
        
        if return_all_losses:
            # 各損失ごとの内訳値もリストとして返す（WandBに個別ロギングするため：LightningModule側で利用）
            return errors, losses
        else:
            return errors


class LossType(enum.Enum):
    """設定ファイルの文字列キーから損失関数クラスを即座にスイッチ・生成するための列挙型マッピングテーブル。"""
    L2LOSS = L2Loss
    COSINEL2LOSS = CosineLoss
    L1SMOOTHLOSS = L1SmoothLoss
    GRADIENTL2LOSS = GradientL2Loss
    CONVECTIONL2LOSS = ConvectionL2Loss
    DIVERGENCEL2LOSS = DivergenceL2Loss
    DIVERGENCEL1LOSS = DivergenceL1Loss
    DIVERGENCEL1SMOOTHLOSS = DivergenceL1SmoothLoss