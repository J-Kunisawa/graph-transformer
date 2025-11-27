from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.h5_dataset import H5Dataset
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.models.processors import (
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)
from graphphysics.models.simulator import Simulator
from graphphysics.utils.loss import LossType, MultiLoss
from graphphysics.utils.nodetype import NodeType


def _safe_get(d: Dict[str, Any], *keys, default=None):
    """ネストされた dict から安全に値を取得。無ければ default を返す。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_preprocessing(
    param: Dict[str, Any],
    device: Union[str, torch.device],
    use_edge_feature: bool = True,
    remove_noise: bool = False,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
) -> Callable[[Data], Data]:
    """
    param から前処理パイプラインを組み立てて返す。
    device は現在使用していないが API 互換のため受け取る（build_preprocessing は CPU 前提）。
    """
    preprocessing_params = param.get("transformations", {}).get("preprocessing", {})
    noise_scale = preprocessing_params.get("noise", 0)
    noise_parameters = None

    if noise_scale and not remove_noise:
        noise_parameters = {
            "noise_index_start": preprocessing_params.get("noise_index_start"),
            "noise_index_end": preprocessing_params.get("noise_index_end"),
            "noise_scale": noise_scale,
            "node_type_index": param["index"]["node_type_index"],
        }

    wpp = param.get("transformations", {}).get("world_pos_parameters", {})
    world_pos_parameters = None
    if isinstance(wpp, dict) and wpp.get("use", False):
        world_pos_parameters = {
            "world_pos_index_start": wpp.get("world_pos_index_start"),
            "world_pos_index_end": wpp.get("world_pos_index_end"),
            "node_type_index": param["index"]["node_type_index"],
            "radius": wpp.get("radius", 0.03),
        }

    return build_preprocessing(
        noise_parameters=noise_parameters,
        world_pos_parameters=world_pos_parameters,
        add_edges_features=use_edge_feature,
        extra_node_features=extra_node_features,
        extra_edge_features=extra_edge_features,
    )


def get_model(param: Dict[str, Any], only_processor: bool = False):
    """
    param に基づきモデル（processor 部分）を返す。
    サポートする型: "epd", "transformer", "transolver"
    """
    model_cfg = param.get("model", {})
    model_type = model_cfg.get("type", "").lower()
    node_input_size = int(model_cfg.get("node_input_size", 0)) + NodeType.SIZE

    if model_type == "epd":
        return EncodeProcessDecode(
            message_passing_num=int(model_cfg.get("message_passing_num", 3)),
            node_input_size=node_input_size,
            edge_input_size=int(model_cfg.get("edge_input_size", 0)),
            output_size=int(model_cfg.get("output_size", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            only_processor=only_processor,
        )
    if model_type == "transformer":
        return EncodeTransformDecode(
            message_passing_num=int(model_cfg.get("message_passing_num", 3)),
            node_input_size=node_input_size,
            output_size=int(model_cfg.get("output_size", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            only_processor=only_processor,
        )
    if model_type == "transolver":
        return TransolverProcessor(
            message_passing_num=int(model_cfg.get("message_passing_num", 3)),
            node_input_size=node_input_size,
            output_size=int(model_cfg.get("output_size", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            num_heads=int(model_cfg.get("num_heads", 2)),
        )

    raise ValueError(f"Model type '{model_type}' not supported.")


def get_simulator(param: Dict[str, Any], model, device: Union[str, torch.device]) -> Simulator:
    """
    Simulator ラッパーを生成して返す。device は文字列 or torch.device を許容。
    """
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    node_input_size = int(param["model"]["node_input_size"]) + NodeType.SIZE

    return Simulator(
        node_input_size=node_input_size,
        edge_input_size=int(param["model"].get("edge_input_size", 0)),
        output_size=int(param["model"]["output_size"]),
        feature_index_start=int(param["index"]["feature_index_start"]),
        feature_index_end=int(param["index"]["feature_index_end"]),
        output_index_start=int(param["index"]["output_index_start"]),
        output_index_end=int(param["index"]["output_index_end"]),
        node_type_index=int(param["index"]["node_type_index"]),
        model=model,
        device=device_t,
    )


def get_dataset(
    param: Dict[str, Any],
    preprocessing: Callable[[Data], Data],
    masking_ratio: Optional[float] = None,
    use_edge_feature: bool = True,
    use_previous_data: bool = False,
    switch_to_val: bool = False,
):
    """
    param に基づいて Dataset インスタンスを返す。
    extension は dataset.extension で指定（"h5" / "xdmf"）。
    """
    dataset_params = param.get("dataset", {})
    targets = dataset_params.get("targets", [])
    if not targets:
        raise ValueError("Please provide a list of target properties to predict.")
    khop = int(dataset_params.get("khop", 1))
    new_edges_ratio = float(dataset_params.get("new_edges_ratio", 0.0))
    extension = dataset_params.get("extension", "").lower()

    world_pos_parameters = None
    if khop > 1:
        transformations = param.get("transformations", {})
        wpp = transformations.get("world_pos_parameters", {})
        if isinstance(wpp, dict) and wpp.get("use", False):
            world_pos_parameters = wpp

    if extension == "h5":
        return H5Dataset(
            h5_path=dataset_params["h5_path"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
            world_pos_parameters=world_pos_parameters,
        )
    if extension == "xdmf":
        return XDMFDataset(
            xdmf_folder=dataset_params["xdmf_folder"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
        )

    raise ValueError(f"Dataset extension '{extension}' not supported.")


def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    """
    データセットの種類に応じた num_workers を返す（将来的に拡張用）。
    """
    extension = param.get("dataset", {}).get("extension", "").lower()
    if extension in ("h5", "xdmf"):
        return int(default_num_workers)
    raise ValueError(f"Dataset extension '{extension}' not supported.")


def get_loss(param: Dict[str, Any], **kwargs) -> Tuple[Any, Union[str, List[str]]]:
    """
    loss 設定を解析して Loss オブジェクトとその名前（または名前リスト）を返す。
    複数指定されていれば MultiLoss を返す。
    """
    loss_cfg = param.get("loss", None)
    if not loss_cfg:
        logger.info("No loss specified, falling back to default L2Loss")
        return LossType.L2LOSS.value(**kwargs), LossType.L2LOSS.name

    types = loss_cfg.get("type", [])
    if not isinstance(types, list) or len(types) == 0:
        raise ValueError("loss.type must be a non-empty list")

    if len(types) > 1:
        losses = [LossType[t.upper()].value(**kwargs) for t in types]
        loss_names = [LossType[t.upper()].name for t in types]
        weights = loss_cfg.get("weights", [1.0] * len(types))
        return MultiLoss(losses, weights), loss_names

    single = LossType[types[0].upper()]
    return single.value(**kwargs), single.name


def get_gradient_method(param: Dict[str, Any], **kwargs) -> Optional[str]:
    """
    loss.gradient_method を返す（存在しなければ None）。
    """
    return _safe_get(param, "loss", "gradient_method", default=None)