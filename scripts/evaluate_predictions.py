#!/usr/bin/env python3
"""
Simple evaluation script to compute rollout RMSE and 1-step RMSE
for a trained checkpoint and parameter JSON.

Usage example:
  python3 scripts/evaluate_predictions.py \
    --params training_config/cylinder_generic.json \
    --ckpt checkpoints/cylinder_epd.ckpt \
    --num-workers 0

This script loads the dataset with the same preprocessing as `predict.py`,
restores the LightningModule from checkpoint and runs a deterministic loop
to compute per-node RMSE (masked according to node types used in loss).
"""
import argparse
import json
import math
import os
from typing import Optional

import torch
from torch_geometric.loader import DataLoader

from loguru import logger

from graphphysics.training.lightning_module import LightningModule, build_mask
from graphphysics.training.parse_parameters import get_dataset, get_preprocessing
from graphphysics.external.aneurysm import build_features


def load_parameters(path: str) -> dict:
    with open(path, "r") as fp:
        return json.load(fp)


def load_lightning_from_ckpt(ckpt_path: str, parameters: dict, dataset) -> LightningModule:
    # prefer Lightning load_from_checkpoint
    traj_len = getattr(dataset, "trajectory_length", getattr(dataset, "traj_len", 599))
    dt = getattr(dataset, "dt", getattr(dataset, "timestep", 1.0))
    try:
        lm = LightningModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            parameters=parameters,
            trajectory_length=int(traj_len),
            timestep=float(dt),
        )
        return lm
    except Exception as e:
        logger.warning(f"load_from_checkpoint failed: {e}; attempting permissive restore")

    # fallback: instantiate with conservative defaults and load state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("state_dict") or ck.get("model") or ck.get("model_state_dict") or ck

    lm = LightningModule(
        parameters=parameters,
        learning_rate=float(parameters.get("training", {}).get("learning_rate", 1e-3)),
        num_steps=int(parameters.get("training", {}).get("num_steps", 100000)),
        warmup=int(parameters.get("training", {}).get("warmup", 1000)),
        trajectory_length=int(traj_len),
        timestep=float(dt),
    )

    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            new_k = k
            if k.startswith("model.") and not k.startswith("model.model."):
                new_k = k[len("model."):]
            cleaned[new_k] = v
        try:
            lm.load_state_dict(cleaned, strict=False)
        except Exception as e:
            logger.warning(f"Failed permissive state load: {e}")
    else:
        try:
            lm.load_state_dict(state, strict=False)
        except Exception as e:
            logger.warning(f"Failed permissive state load: {e}")

    return lm


def evaluate(ckpt_path: str, params_path: str, batch_size: int = 1, num_workers: int = 0, use_edge_feature: bool = True):
    parameters = load_parameters(params_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True,
        extra_node_features=build_features,
    )

    dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=True,
    )

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    lm = load_lightning_from_ckpt(ckpt_path, parameters, dataset)
    lm.to(device)
    lm.eval()

    total_sq = 0.0
    total_count = 0
    first_step_rmses = []

    last_pred = None
    last_prev_pred = None
    current_traj = None
    step_counter = 0

    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)

            # detect trajectory boundary (traj_index may be a 1-element tensor)
            traj_idx = getattr(batch, "traj_index", None)
            if traj_idx is not None:
                try:
                    if hasattr(traj_idx, "item"):
                        traj_val = int(traj_idx.item())
                    else:
                        traj_val = int(traj_idx)
                except Exception:
                    traj_val = None
            else:
                traj_val = None

            if traj_val is not None and (current_traj is None or traj_val > current_traj):
                # new trajectory -> reset counters
                current_traj = traj_val
                step_counter = 0
                last_pred = None
                last_prev_pred = None

            # use LightningModule helper to make a single prediction step
            batch_out, predicted_outputs, target, last_pred, last_prev_pred = lm._make_prediction(
                batch, last_pred, last_prev_pred
            )

            # compute include mask as inverse of build_mask
            masked = build_mask(lm.param, batch)
            include_mask = torch.logical_not(masked)

            errs = ((predicted_outputs - target) ** 2)[include_mask]
            total_sq += float(errs.sum().item()) if errs.numel() > 0 else 0.0
            total_count += int(errs.numel())

            if step_counter == 0:
                if errs.numel() > 0:
                    first_rmse = math.sqrt(float(errs.mean().item()))
                else:
                    first_rmse = 0.0
                first_step_rmses.append(first_rmse)

            step_counter += 1

    overall_rmse = math.sqrt(total_sq / total_count) if total_count > 0 else float("nan")
    mean_first_step_rmse = float(sum(first_step_rmses) / len(first_step_rmses)) if first_step_rmses else float("nan")

    print("Evaluation results:")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  params: {params_path}")
    print(f"  examples (batches): {len(dataset)}")
    print(f"  overall RMSE (all rollout): {overall_rmse:.6f}")
    print(f"  mean 1-step RMSE: {mean_first_step_rmse:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="Path to training parameters JSON")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-edge-feature", action="store_true", help="Disable edge features in preprocessing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt_path=args.ckpt,
        params_path=args.params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_edge_feature=not args.no_edge_feature,
    )
