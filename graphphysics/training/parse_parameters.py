import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import MultiStepDataset
from graphphysics.dataset.h5_dataset import H5Dataset
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.models.layers import (
    set_memory_optimized_training,
    set_use_silu_activation,
)
from graphphysics.models.processors import (
    EagleProcessor,
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)
from graphphysics.models.simulator import Simulator
from graphphysics.utils.loss import LossType, MultiLoss
from graphphysics.utils.nodetype import NodeType


def get_preprocessing(
    param: Dict[str, Any],
    device: torch.device,
    use_edge_feature: bool = True,
    remove_noise: bool = False,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
):
    """
    設定パラメータに基づいてグラフの「前処理・データ拡張パイプライン（T.Compose）」を動的に構築する。
    ガウスノイズパラメータや、障害物・流体間の相互作用を解くための絶対座標パラメータをパースする。
    """
    # 1. カリキュラム学習やロバスト性向上のためのガウスノイズパラメータの抽出
    preprocessing_params = param.get("transformations", {}).get("preprocessing", {})
    noise_scale = preprocessing_params.get("noise", 0)
    noise_parameters = None

    if noise_scale != 0 and not remove_noise:
        noise_parameters = {
            "noise_index_start": preprocessing_params.get("noise_index_start"),
            "noise_index_end": preprocessing_params.get("noise_index_end"),
            "noise_scale": noise_scale,
            "node_type_index": param["index"]["node_type_index"], # 壁等の境界を除外するためのインデックス
        }

    # 2. 移動する障害物等との幾何関係を解くための絶対座標パラメータの抽出
    world_pos_params = param.get("transformations", {}).get("world_pos_parameters", {})
    world_pos_parameters = None
    if world_pos_params.get("use", False):
        world_pos_parameters = {
            "world_pos_index_start": world_pos_params.get("world_pos_index_start"),
            "world_pos_index_end": world_pos_params.get("world_pos_index_end"),
            "node_type_index": param["index"]["node_type_index"],
        }

    # 共通前処理モジュール (preprocessing.py) の build_preprocessing 関数へ丸投げして統合
    return build_preprocessing(
        noise_parameters=noise_parameters,
        world_pos_parameters=world_pos_parameters,
        add_edges_features=use_edge_feature,
        extra_node_features=extra_node_features,
        extra_edge_features=extra_edge_features,
    )


def get_model(param: Dict[str, Any], only_processor: bool = False):
    """
    設定JSONのパラメータを解析し、指定された演算コアモデル（GNN / Transformer / Transolver）を
    適切な内部パラメータ、活性化関数、省メモリモードに同期させた上でインスタンス化して返す。
    """
    model_type = param.get("model", {}).get("type", "")
    
    # 物理特徴量の次元数に、ノード属性のワンホットベクトル（NodeType.SIZE）の次元を自動加算結合
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE
    
    # 各種設定フラグの抽出
    use_silu = param.get("model", {}).get("use_silu_activation", False)
    training_params = param.get("training", {})
    enable_vram_optimizations = training_params.get("enable_vram_optimizations", False)
    use_rope = param.get("model", {}).get("use_rope_embeddings", False)
    use_gated_attention = param.get("model", {}).get("use_gated_attention", False)
    use_gated_mlp = param.get("model", {}).get("use_gated_mlp", False)
    use_temporal_block = training_params.get("use_temporal_block", False)
    rope_pos_dimension = param.get("model", {}).get("rope_pos_dimension", 3)
    rope_base = param.get("model", {}).get("rope_base", 10000.0)
    rope_length_scale = 1.0
    if use_rope:
        meta_path = param.get("dataset", {}).get("meta_path")
        if meta_path is None:
            raise ValueError(
                "RoPEの無次元化に必要な dataset.meta_path が設定されていません。"
            )
        with open(meta_path, "r") as fp:
            dataset_meta = json.load(fp)
        rope_length_scale = dataset_meta.get("geometry", {}).get(
            "obstacle_diameter"
        )
        if rope_length_scale is None:
            raise ValueError(
                "use_rope_embeddings=True の場合、meta.json に "
                "geometry.obstacle_diameter を設定してください。"
            )
        rope_length_scale = float(rope_length_scale)
        if rope_length_scale <= 0:
            raise ValueError(
                "geometry.obstacle_diameter は正の値である必要があります。"
            )
    dropout = param.get("model", {}).get(
        "dropout", param.get("model", {}).get("dropout_rate", 0.0)
    )
    mlp_ratio = param.get("model", {}).get("mlp_ratio", 1)
    slice_num = param.get("model", {}).get("slice_num", 32)
    ref = param.get("model", {}).get("ref", 8)
    unified_pos = param.get("model", {}).get("unified_pos", False)
    adjacency_augmentation = param.get("model", {}).get(
        "adjacency_augmentation", param.get("adjacency_augmentation", {})
    )
    
    # グローバル共通レイヤー(layers.py)の状態フラグを瞬時に書き換えて同期
    set_use_silu_activation(use_silu)
    set_memory_optimized_training(enable_vram_optimizations)

    # --- アーキテクチャの種類によるファクトリ分岐 ---
    if model_type == "epd":
        # メッセージパッシングGNN (MeshGraphNetスタイル)
        return EncodeProcessDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            edge_input_size=param["model"]["edge_input_size"],
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            only_processor=only_processor,
            use_rope_embeddings=use_rope,
            use_gated_attention=use_gated_attention,
            use_gated_mlp=use_gated_mlp,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_temporal_block=use_temporal_block,
        )
    elif model_type == "transformer":
        # 自己アテンション空間モデル (DGL Sparse / PyG TransformerConv)
        return EncodeTransformDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            num_heads=param["model"]["num_heads"],
            only_processor=only_processor,
            use_rope_embeddings=use_rope,
            use_gated_attention=use_gated_attention,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            rope_length_scale=rope_length_scale,
            use_temporal_block=use_temporal_block,
            adjacency_augmentation=adjacency_augmentation,
        )
    elif model_type == "transolver":
        # 統一幾何グリッド・スライストークンPDE基盤モデル (Transolver++)
        return TransolverProcessor(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            num_heads=param["model"]["num_heads"],
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            ref=ref,
            unified_pos=unified_pos,
            use_rope_embeddings=use_rope,
            use_gated_attention=use_gated_attention,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_temporal_block=use_temporal_block,
        )
    elif model_type == "eagle":
        # EAGLE: cluster pooling -> global attention -> graph retrieval
        model_params = param["model"]
        return EagleProcessor(
            message_passing_num=model_params["message_passing_num"],
            node_input_size=node_input_size,
            output_size=model_params["output_size"],
            hidden_size=model_params.get("hidden_size", 128),
            num_heads=model_params.get("num_heads", 4),
            dropout=dropout,
            token_size=model_params.get("token_size"),
            cluster_size=model_params.get("cluster_size", 20),
            space_dim=model_params.get("space_dim", 2),
            eagle_gnn_layers=model_params.get("eagle_gnn_layers", 4),
            pos_frequencies=model_params.get("pos_frequencies", 8),
            pos_frequency_start=model_params.get("pos_frequency_start", -3),
        )
    else:
        raise ValueError(f"モデルタイプ '{model_type}' はサポートされていません。")


def get_simulator(param: Dict[str, Any], model, device: torch.device) -> Simulator:
    """
    構築された演算コアモデルをラップし、データの時間積分・残差処理・標準化を制御する
    統合シミュレータ環境 (Simulator) を生成する。
    """
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE

    return Simulator(
        node_input_size=node_input_size,
        edge_input_size=param["model"]["edge_input_size"],
        output_size=param["model"]["output_size"],
        feature_index_start=param["index"]["feature_index_start"],
        feature_index_end=param["index"]["feature_index_end"],
        output_index_start=param["index"]["output_index_start"],
        output_index_end=param["index"]["output_index_end"],
        node_type_index=param["index"]["node_type_index"],
        model=model,
        device=device,
    )


def get_dataset(
    param: Dict[str, Any],
    preprocessing: Callable[[Data], Data],
    masking_ratio: Optional[float] = None,
    use_edge_feature: bool = True,
    use_previous_data: bool = False,
    switch_to_val: bool = False,
    use_test: bool = False,
    use_partitioning: bool = False,
    num_partitions: int = None,
    max_nodes_per_partition: int = None,
    multi_step_steps: int = 1,
):
    """
    設定パラメータに準拠したデータセットインスタンスを構築する。
    ファイル形式のパース、K-hop長距離接続の幾何属性判定、訓練/テストデータのパスの安全切り替えを担う。
    """
    dataset_params = param.get("dataset", {})
    targets = dataset_params.get("targets", [])
    if len(targets) == 0:
        raise ValueError("モデルが予測すべきターゲット属性リスト（targets）を1つ以上指定してください。")
        
    khop = dataset_params.get("khop", 1)
    new_edges_ratio = dataset_params.get("new_edges_ratio", 0)
    extension = dataset_params.get("extension", "")
    train_path = dataset_params["train_path"]
    val_path = dataset_params["val_path"]
    test_path = dataset_params["test_path"]

    if test_path == train_path:
        raise ValueError(f"訓練用パスとテスト用パスに、全く同じファイルパス（ {train_path} ）を指定することはできません。")

    # 選択ルール:
    # - 明示的に use_test=True の場合は test_path を使用
    # - それ以外で switch_to_val=True の場合は val_path を使用
    # - それ以外は train_path を使用
    if use_test:
        dataset_path = test_path
        meta_key = "test_meta_path"
    elif switch_to_val:
        dataset_path = val_path
        meta_key = "val_meta_path"
    else:
        dataset_path = train_path
        meta_key = "train_meta_path"

    # 分割ごとのメタファイルを優先し、未指定なら従来の共通設定を使用する。
    meta_path = dataset_params.get(meta_key) or dataset_params.get("meta_path")
    if not meta_path:
        raise ValueError(f"dataset.{meta_key} または dataset.meta_path を指定してください。")

    # K-hop長距離通信時、絶対座標に基づく特徴生成を行うか幾何判定
    world_pos_parameters = None
    if khop > 1:
        transformations = param.get("transformations", {})
        if "world_pos_parameters" in transformations:
            wpp = transformations["world_pos_parameters"]
            if wpp.get("use", False):
                world_pos_parameters = wpp

    # --- 拡張子による具体的なデータセットクラスへの分岐マッピング ---
    if extension == "h5":
        dataset = H5Dataset(
            h5_path=dataset_path, meta_path=meta_path, targets=targets,
            preprocessing=preprocessing, masking_ratio=masking_ratio, khop=khop,
            new_edges_ratio=new_edges_ratio, add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data, world_pos_parameters=world_pos_parameters,
            use_partitioning=use_partitioning, num_partitions=num_partitions,
            max_nodes_per_partition=max_nodes_per_partition,
        )
    elif extension == "xdmf":
        dataset = XDMFDataset(
            xdmf_folder=dataset_path, meta_path=meta_path, targets=targets,
            preprocessing=preprocessing, masking_ratio=masking_ratio, khop=khop,
            new_edges_ratio=new_edges_ratio, add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data, use_partitioning=use_partitioning,
            num_partitions=num_partitions, max_nodes_per_partition=max_nodes_per_partition,
        )
    else:
        raise ValueError(f"データセットの拡張子形式 '{extension}' はサポートされていません。")

    if multi_step_steps > 1:
        return MultiStepDataset(dataset, steps=multi_step_steps)
    return dataset

def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    """
    データファイル形式の特性に応じて、DataLoaderに割り当てる最適な並列CPU子プロセス（ワーカー）数を決定する。
    (将来的なH5固有のロック競合調整などのための拡張用プレースホルダー)
    """
    dataset_params = param.get("dataset", {})
    extension = dataset_params.get("extension", "")
    if extension in {"h5", "xdmf"}:
        return default_num_workers
    else:
        raise ValueError(f"データセットの拡張子形式 '{extension}' はサポートされていません。")


def get_loss(param: Dict[str, Any], **kwargs):
    """
    損失関数を初期化パースする。
    複数の損失が指定されている場合は、重み（weights）をブレンドした MultiLoss オブジェクトを全自動生成する。
    """
    try:
        _ = param["loss"]
    except KeyError:
        logger.info("損失関数（loss）の指定が無いため、デフォルトの L2Loss（MSE）へフォールバックします。")
        return LossType.L2LOSS.value(**kwargs), LossType.L2LOSS.name

    # リスト内に複数のロスタイプが指定されている場合（例：L2誤差 ＋ 勾配誤差 ＋ 発散誤差 = 物理情報駆動）
    if len(param["loss"]["type"]) > 1:
        losses = [LossType[t.upper()].value(**kwargs) for t in param["loss"]["type"]]
        losses_names = [LossType[t.upper()].name for t in param["loss"]["type"]]
        weights = param["loss"]["weights"] # 各ロスの重み（ブレンド比率）
        
        # 複合マルチロスオブジェクトとしてカプセル化して返す
        return MultiLoss(losses, weights), losses_names
    else:
        # 単一ロスの通常モード
        loss = LossType[param["loss"]["type"][0].upper()]
        return loss.value(**kwargs), loss.name


def get_gradient_method(param: Dict[str, Any], **kwargs) -> str:
    """
    メッシュ上で物理情報損失（PINNロス）を解く際の、空間数値微分手法（有限差分か最小二乗か）の設定をパースする。
    """
    try:
        gradient_method = param["loss"]["gradient_method"]
    except KeyError:
        logger.info("空間微分計算法（gradient_method）の指定はありません。")
        gradient_method = None
    return gradient_method
