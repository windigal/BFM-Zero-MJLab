# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

from humanoidverse.agents.envs.expert_motion_loader import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluation,
    HumanoidVerseMjlabTrackingEvaluationConfig,
)

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import math
import time
import typing as tp
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import wandb
import safetensors.torch
from packaging.version import Version
from torch.utils._pytree import tree_map
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.tldr_dist_aux.agent import TldrDistAuxAgentConfig
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere
from humanoidverse.distributed import (
    average_metrics,
    barrier,
    broadcast_agent_state,
    broadcast_object,
    module_sync_report,
    sync_floating_buffers,
)

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"
B_CONVERGENCE_PROBE_FILENAME = "b_convergence_probe.pt"
EMD_GROUP_NAMES = ("non_high", "high")
PACE_GROUP_NAMES = ("mastered", "challenging")

TRAIN_METRIC_GROUPS = OrderedDict(
    [
        (
            "FB",
            [
                "B",
                "B_norm",
                "z_norm",
                "F1",
                "M1",
                "target_M",
                "fb_diag",
                "fb_offdiag",
                "orth_loss_diag",
                "orth_loss_offdiag",
                "orth_loss",
                "q_loss",
                "fb_loss",
            ],
        ),
        (
            "TeCH",
            [
                "tldr_te_loss",
                "tldr_te_obj_mean",
                "tldr_dist_mean",
                "tldr_cst_penalty_mean",
                "tldr_dual_lam",
                "tldr_dual_loss",
                "tldr_soft_term",
            ],
        ),
        (
            "GCR",
            [
                "gcr_contrastive_loss_raw",
                "gcr_contrastive_loss",
                "gcr_poincare_pos_dist",
                "gcr_cos_pos",
            ],
        ),
        (
            "Discriminator",
            [
                "disc_expert_loss",
                "disc_train_loss",
                "disc_wgan_gp_loss",
                "disc_expert_logit",
                "disc_train_logit",
                "disc_loss",
                "mean_disc_reward",
            ],
        ),
        ("Critic", ["target_Q", "Q1", "mean_next_Q", "unc_Q", "critic_loss", "mean_reward_train"]),
        (
            "Aux Critic",
            ["target_auxQ", "auxQ1", "mean_next_auxQ", "unc_auxQ", "aux_critic_loss", "mean_aux_reward"],
        ),
        (
            "Actor",
            [
                "Q_discriminator",
                "Q_actor",
                "Q_aux",
                "Q_fb",
                "q",
                "disc_reward_actor",
                "actor_loss",
            ],
        ),
        (
            "Tracking Q",
            [
                "tracking_q/target",
                "tracking_q/value",
                "tracking_q/next_value",
                "tracking_q/uncertainty",
                "tracking_q/critic_loss",
                "tracking_q/reward",
                "tracking_q/actor_q",
                "tracking_q/actor_uncertainty",
                "tracking_q/actor_loss",
                "tracking_q/base_actor_loss",
                "tracking_q/actor_coefficient",
                "tracking_q/grad_scale",
                "tracking_q/grad_base_ema",
                "tracking_q/grad_tracking_ema",
                "tracking_q/weighted_grad_ratio",
                "tracking_q/grad_cosine",
                "tracking_q/actor_ramp",
                "tracking_q/expert_batch_fraction",
                "tracking_q/grad_norm_tracking_raw",
                "tracking_q/grad_norm_tracking_weighted",
                "tracking_q/grad_norm_discriminator",
                "tracking_q/grad_norm_auxiliary",
                "tracking_q/grad_norm_fb",
                "tracking_q/grad_norm_consolidation",
                "tracking_q/grad_cosine_discriminator_vs_tracking",
                "tracking_q/grad_cosine_auxiliary_vs_tracking",
                "tracking_q/grad_cosine_fb_vs_tracking",
                "tracking_q/grad_cosine_consolidation_vs_tracking",
                "tracking_q/conflict_projection_discriminator_vs_tracking",
                "tracking_q/conflict_projection_auxiliary_vs_tracking",
                "tracking_q/conflict_projection_fb_vs_tracking",
                "tracking_q/conflict_projection_consolidation_vs_tracking",
            ],
        ),
        (
            "Goal Reach",
            [
                "goal_reach/probe_envs_per_rank",
                "goal_reach/completed_goals",
                "goal_reach/interrupted_goals",
                "goal_reach/close_rate",
                "goal_reach/plateau_rate",
                "goal_reach/arrival_rate",
                "goal_reach/close_step_mean",
                "goal_reach/plateau_step_mean",
                "goal_reach/arrival_step_mean",
                "goal_reach/arrival_step_capped_mean",
                "goal_reach/z_cosine_initial",
                "goal_reach/z_cosine_final",
                "goal_reach/z_cosine_best",
                "goal_reach/z_rmse_initial",
                "goal_reach/z_rmse_final",
                "goal_reach/z_rmse_best",
            ],
        ),
        (
            "Tracking Rewards",
            [
                "tracking_reward/anchor_orientation",
                "tracking_reward/relative_body_position",
                "tracking_reward/relative_body_orientation",
                "tracking_reward/global_body_linear_velocity",
                "tracking_reward/global_body_angular_velocity",
                "tracking_reward/total",
            ],
        ),
        (
            "PACE",
            [
                "pace/lambda_con",
                "pace/rho",
                "pace/rho_ema",
                "pace/valid_acquisition",
                "pace/valid_consolidation",
                "pace/valid_ratio_acquisition",
                "pace/valid_ratio_consolidation",
                "pace/batch_acquisition",
                "pace/batch_consolidation",
                "pace/windows_completed",
                "pace/loss_acquisition",
                "pace/loss_acquisition_base",
                "pace/loss_acquisition_weighted",
                "pace/loss_consolidation",
                "pace/loss_consolidation_weighted",
                "pace/actor_grad_norm_acquisition",
                "pace/actor_grad_norm_consolidation",
                "pace/actor_grad_norm_consolidation_weighted",
                "pace/actor_grad_ratio_consolidation_to_acquisition",
                "pace/actor_grad_cosine",
                "pace/consolidation_scale",
                "pace/effective_consolidation_coefficient",
                "pace/consolidation_action_rmse",
                "pace/actor_update_applied",
                "pace/actor_lr",
                "pace/actor_lr_multiplier",
                "pace/actor_warmup_active",
                "pace/acquisition_loss_weight",
                "pace/retention_warning",
                "pace/retention_recovery",
            ],
        ),
        ("Rewards", ["mean_env_reward", "mean_total_reward"]),
        (
            "Aux Rewards",
            [
                "aux_rew/penalty_torques",
                "aux_rew/penalty_action_rate",
                "aux_rew/limits_dof_pos",
                "aux_rew/limits_torque",
                "aux_rew/penalty_undesired_contact",
                "aux_rew/penalty_feet_ori",
                "aux_rew/penalty_ankle_roll",
                "aux_rew/penalty_slippage",
            ],
        ),
        ("Latent", ["z_norm/train", "z_norm/expert", "z_norm/mixed"]),
        ("Weights", ["weight/disc_actor", "weight/aux_actor", "weight/scale_q", "weight/disc_reward_coef"]),
        (
            "Config",
            [
                "cfg/reg_coeff_disc",
                "cfg/reg_coeff_aux",
                "cfg/disc_reward_coef",
                "cfg/train_goal_ratio",
                "cfg/expert_asm_ratio",
                "cfg/relabel_ratio",
            ],
        ),
        (
            "Environment",
            [
                "torso_contact_force_mean",
                "torso_contact_force_max",
                "torso_contact_force_p95",
                "torso_contact_force_contact_count",
                "torso_contact_force_contact_frac",
                "torso_contact_force_mean_x",
                "torso_contact_force_mean_y",
                "torso_contact_force_mean_z",
                "torso_contact_force_z_mean",
                "torso_contact_force_z_max",
                "torso_contact_force_z_p95",
                "torso_z_mean",
                "torso_z_min",
                "torso_z_p05",
                "torso_z_contact_mean",
                "torso_z_contact_min",
                "torso_z_contact_p05",
            ],
        ),
        (
            "Distributed",
            [
                "distributed/world_size",
                "distributed/effective_batch_size",
                "distributed/num_envs_per_rank",
                "distributed/global_parallel_envs",
                "distributed/replay_capacity_per_rank",
                "distributed/effective_replay_capacity",
                "distributed/trajectory_steps_per_rank",
                "distributed/compile",
                "distributed/local_env_steps",
                "distributed/global_env_steps",
                "distributed/optimizer_steps",
            ],
        ),
    ]
)


def _format_metric_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = format(Decimal(str(float(value))), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _as_python_scalar(value: tp.Any) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return None


def _build_grouped_metrics(metrics: dict[str, tp.Any]) -> OrderedDict[str, list[tuple[str, float | int]]]:
    scalar_metrics = {}
    for key, value in metrics.items():
        scalar = _as_python_scalar(value)
        if scalar is not None:
            scalar_metrics[key] = scalar

    grouped = OrderedDict()
    used = set()
    for group_name, keys in TRAIN_METRIC_GROUPS.items():
        items = [(key, scalar_metrics[key]) for key in keys if key in scalar_metrics]
        if items:
            grouped[group_name] = items
            used.update(key for key, _ in items)
    leftover = [(key, scalar_metrics[key]) for key in sorted(scalar_metrics.keys()) if key not in used]
    if leftover:
        grouped["Other"] = leftover
    return grouped


def _iter_tracked_group_metrics(
    grouped_metrics: OrderedDict[str, list[tuple[str, float | int]]],
) -> tp.Iterator[tuple[str, str, float | int]]:
    for group_name in TRAIN_METRIC_GROUPS:
        for key, value in grouped_metrics.get(group_name, []):
            yield group_name, key, value


def _render_grouped_train_log(
    *,
    grouped_metrics: OrderedDict[str, list[tuple[str, float | int]]],
    timestep: int,
    iteration: int,
    max_iterations: int,
    steps_per_second: float,
    collection_time: float,
    learning_time: float,
    elapsed_time: float,
    eta_seconds: float,
) -> str:
    width = 88
    summary_labels = ["Computation", "Total timesteps", "Iteration time", "Time elapsed", "ETA"]
    metric_labels = [
        key if "/" in key else f"{group_name}/{key}"
        for group_name, items in grouped_metrics.items()
        for key, _ in items
    ]
    label_width = max([len(label) for label in metric_labels + summary_labels], default=0)
    lines = [
        "#" * width,
        f"{'Learning iteration ' + str(iteration) + '/' + str(max_iterations):^{width}}",
        "",
        (
            f"{'Computation':>{label_width}}: "
            f"{_format_metric_value(steps_per_second)} steps/s "
            f"(collection: {collection_time:.3f}s, learning: {learning_time:.3f}s)"
        ),
    ]
    for group_name, items in grouped_metrics.items():
        for key, value in items:
            metric_label = key if "/" in key else f"{group_name}/{key}"
            lines.append(f"{metric_label:>{label_width}}: {_format_metric_value(value)}")
    lines.extend(
        [
            "-" * width,
            f"{'Total timesteps':>{label_width}}: {timestep}",
            f"{'Iteration time':>{label_width}}: {collection_time + learning_time:.2f}s",
            f"{'Time elapsed':>{label_width}}: {_format_duration(elapsed_time)}",
            f"{'ETA':>{label_width}}: {_format_duration(eta_seconds)}",
        ]
    )
    return "\n".join(lines)


def _tensorboard_tag(group_name: str, key: str) -> str:
    group_slug = group_name.lower().replace(" ", "_")
    known_prefixes = {
        "Aux Rewards": "aux_rew/",
        "Latent": "z_norm/",
        "Weights": "weight/",
        "Config": "cfg/",
        "Distributed": "distributed/",
        "PACE": "pace/",
        "Tracking Q": "tracking_q/",
        "Tracking Rewards": "tracking_reward/",
        "Goal Reach": "goal_reach/",
    }
    prefix = known_prefixes.get(group_name)
    tag_key = key.removeprefix(prefix) if prefix else key
    return f"train/{group_slug}/{tag_key}"


def _compute_emd_group_distribution(
    evaluation_metrics: dict,
    expected_motion_ids: list[int],
    *,
    low_threshold: float,
    high_threshold: float,
    scale: float,
    high_group_sampling_ratio: float,
    high_group_fraction: float | None = None,
) -> dict[str, tp.Any]:
    """Build the two-source single-dataset EMD distribution from one complete eval."""
    if not np.isfinite(low_threshold) or not np.isfinite(high_threshold) or not 0 <= low_threshold < high_threshold:
        raise ValueError(
            f"Invalid EMD group thresholds: require 0 <= low < high, got low={low_threshold}, high={high_threshold}"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"EMD prioritization scale must be finite and positive, got {scale}")
    if not np.isfinite(high_group_sampling_ratio) or not 0 < high_group_sampling_ratio < 1:
        raise ValueError(
            "High-EMD sampling ratio must be finite and strictly between 0 and 1, "
            f"got {high_group_sampling_ratio}"
        )
    if high_group_fraction is not None and (
        not np.isfinite(high_group_fraction) or not 0 < high_group_fraction < 1
    ):
        raise ValueError(
            "High-EMD group fraction must be finite and strictly between 0 and 1, "
            f"got {high_group_fraction}"
        )

    expected = [int(motion_id) for motion_id in expected_motion_ids]
    if len(expected) == 0 or len(set(expected)) != len(expected):
        raise ValueError("Expected motion ids must be non-empty and unique")
    if expected != list(range(len(expected))):
        raise ValueError(
            "Single-source EMD grouping requires expert motion ids to match global motion-library order 0..N-1"
        )
    expected_set = set(expected)
    emd_by_id: dict[int, float] = {}
    for motion_name, metric in evaluation_metrics.items():
        if "motion_id" not in metric or "emd" not in metric:
            raise ValueError(f"Tracking eval entry {motion_name!r} is missing motion_id or emd")
        motion_id = int(metric["motion_id"])
        if motion_id in emd_by_id:
            raise ValueError(f"Tracking eval returned duplicate motion_id={motion_id} ({motion_name!r})")
        emd = float(metric["emd"])
        if not np.isfinite(emd):
            raise ValueError(f"Tracking eval returned non-finite EMD for motion_id={motion_id}, motion={motion_name!r}: {emd}")
        emd_by_id[motion_id] = emd

    actual_set = set(emd_by_id)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        raise ValueError(f"Tracking eval motion ids do not match expert data: missing={missing}, unexpected={unexpected}")

    emd = torch.tensor([emd_by_id[motion_id] for motion_id in expected], dtype=torch.float64)
    # Low and mid EMD trajectories are one source; the high source is defined
    # solely by the upper threshold. The lower threshold remains the minimum
    # clamp used by the existing exponential within-source priority formula.
    if high_group_fraction is None:
        group_ids = (emd >= high_threshold).to(torch.long)
        high_group_cutoff = float(high_threshold)
    else:
        # Select an exact, deterministic top fraction by EMD. Stable sorting
        # breaks boundary ties by global motion id, so every rank reconstructs
        # the same group and a 5000-motion dataset gets exactly 500 high items
        # for a requested fraction of 0.10.
        high_count = min(max(int(round(len(expected) * high_group_fraction)), 1), len(expected) - 1)
        high_ids = torch.argsort(emd, descending=True, stable=True)[:high_count]
        group_ids = torch.zeros(len(expected), dtype=torch.long)
        group_ids[high_ids] = 1
        high_group_cutoff = float(emd[high_ids].min().item())
    counts = torch.bincount(group_ids, minlength=2)
    data_ratios = counts.to(torch.float64) / len(expected)
    requested_group_weights = torch.tensor(
        [1.0 - high_group_sampling_ratio, high_group_sampling_ratio],
        dtype=torch.float64,
    )
    # A fixed share cannot be assigned to an empty source. Renormalize over
    # sources that contain data so training can continue after a future eval in
    # which every trajectory falls on the same side of the threshold.
    group_weights = requested_group_weights * (counts > 0)
    group_weights = group_weights / group_weights.sum()
    probabilities = torch.zeros(len(expected), dtype=torch.float64)
    clamped = torch.clamp(emd, min=low_threshold, max=high_threshold)
    logits = np.log(2.0) * float(scale) * clamped
    stats: dict[str, float | int] = {}
    for group_id, group_name in enumerate(EMD_GROUP_NAMES):
        mask = group_ids == group_id
        count = int(counts[group_id].item())
        if count:
            probabilities[mask] = group_weights[group_id] * torch.softmax(logits[mask], dim=0)
            mean_emd = float(emd[mask].mean().item())
        else:
            mean_emd = 0.0
        stats[f"eval/emd_sampling/{group_name}/data_ratio"] = float(data_ratios[group_id].item())
        stats[f"eval/emd_sampling/{group_name}/sampling_ratio"] = float(group_weights[group_id].item())
        stats[f"eval/emd_sampling/{group_name}/count"] = count
        stats[f"eval/emd_sampling/{group_name}/mean_emd"] = mean_emd
    stats["eval/emd_sampling/high/cutoff_emd"] = high_group_cutoff
    stats["eval/emd_sampling/high/requested_data_fraction"] = (
        float(high_group_fraction) if high_group_fraction is not None else float(data_ratios[1].item())
    )
    if not torch.isfinite(probabilities).all() or not torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise RuntimeError("Computed EMD trajectory probabilities are not a finite normalized distribution")
    return {
        "probabilities": probabilities.to(torch.float32),
        "group_ids": group_ids,
        "group_weights": group_weights.to(torch.float32),
        "stats": stats,
    }


def _compute_pace_emd_distribution(
    evaluation_metrics: dict,
    expected_motion_ids: list[int],
    *,
    challenging_fraction: float,
    challenging_sampling_ratio: float,
    clamp_min: float,
    clamp_max: float,
    scale: float,
    fixed_group_ids: torch.Tensor | list[int] | None = None,
    challenging_weight_mode: tp.Literal["absolute", "rank"] = "rank",
) -> dict[str, tp.Any]:
    """Build PACE sampling weights while keeping the base-policy split immutable."""
    if not np.isfinite(challenging_fraction) or not 0 < challenging_fraction < 1:
        raise ValueError("PACE challenging_fraction must be strictly between 0 and 1")
    if not np.isfinite(challenging_sampling_ratio) or not 0 < challenging_sampling_ratio < 1:
        raise ValueError("PACE challenging_sampling_ratio must be strictly between 0 and 1")
    if not np.isfinite(clamp_min) or not np.isfinite(clamp_max) or not 0 <= clamp_min < clamp_max:
        raise ValueError("PACE EMD clamps must satisfy 0 <= min < max")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("PACE EMD scale must be finite and positive")
    if challenging_weight_mode not in ("absolute", "rank"):
        raise ValueError("PACE challenging_weight_mode must be 'absolute' or 'rank'")

    expected = [int(motion_id) for motion_id in expected_motion_ids]
    if len(expected) < 2 or expected != list(range(len(expected))):
        raise ValueError("PACE requires expert motion ids in global order 0..N-1")
    emd_by_id: dict[int, float] = {}
    for motion_name, metric in evaluation_metrics.items():
        if "motion_id" not in metric or "emd" not in metric:
            raise ValueError(f"Tracking eval entry {motion_name!r} is missing motion_id or emd")
        motion_id = int(metric["motion_id"])
        if motion_id in emd_by_id:
            raise ValueError(f"Tracking eval returned duplicate motion_id={motion_id}")
        emd = float(metric["emd"])
        if not np.isfinite(emd):
            raise ValueError(f"Tracking eval returned non-finite EMD for motion_id={motion_id}: {emd}")
        emd_by_id[motion_id] = emd
    if set(emd_by_id) != set(expected):
        missing = sorted(set(expected) - set(emd_by_id))
        unexpected = sorted(set(emd_by_id) - set(expected))
        raise ValueError(f"PACE eval motion ids do not match expert data: missing={missing}, unexpected={unexpected}")

    emd = torch.tensor([emd_by_id[motion_id] for motion_id in expected], dtype=torch.float64)
    challenging_count = min(max(int(math.ceil(len(expected) * challenging_fraction)), 1), len(expected) - 1)
    if fixed_group_ids is None:
        # The first full eval uses the immutable base policy. Stable sorting makes
        # equal-EMD ties deterministic in global motion-id order.
        order = torch.argsort(emd, descending=True, stable=True)
        group_ids = torch.zeros(len(expected), dtype=torch.long)
        group_ids[order[:challenging_count]] = 1
    else:
        group_ids = torch.as_tensor(fixed_group_ids, dtype=torch.long, device="cpu").reshape(-1).clone()
        if len(group_ids) != len(expected):
            raise ValueError(
                f"PACE fixed group ids have length {len(group_ids)}, expected {len(expected)}"
            )
        if not torch.all((group_ids == 0) | (group_ids == 1)):
            raise ValueError("PACE fixed group ids must contain only 0 (mastered) and 1 (challenging)")
        if int((group_ids == 1).sum().item()) != challenging_count:
            raise ValueError(
                "PACE fixed challenging group size no longer matches challenging_fraction: "
                f"actual={int((group_ids == 1).sum().item())}, expected={challenging_count}"
            )
    group_weights = torch.tensor(
        [1.0 - challenging_sampling_ratio, challenging_sampling_ratio], dtype=torch.float64
    )
    probabilities = torch.zeros(len(expected), dtype=torch.float64)
    mastered = group_ids == 0
    challenging = group_ids == 1
    probabilities[mastered] = group_weights[0] / mastered.sum()
    if challenging_weight_mode == "absolute":
        priority_values = torch.clamp(emd[challenging], min=clamp_min, max=clamp_max)
    else:
        # Absolute EMDs saturated at clamp_max in the failed run. Rank-relative
        # priorities retain the original 2^(scale*x) shape without collapsing
        # almost every fixed challenging trajectory to the same maximum value.
        challenging_emd = emd[challenging]
        rank_order = torch.argsort(challenging_emd, stable=True)
        priority_values = torch.empty_like(challenging_emd)
        priority_values[rank_order] = torch.linspace(
            clamp_min,
            clamp_max,
            steps=len(challenging_emd),
            dtype=torch.float64,
        )
    logits = np.log(2.0) * float(scale) * priority_values
    probabilities[challenging] = group_weights[1] * torch.softmax(logits, dim=0)

    stats: dict[str, float | int] = {
        "eval/pace_sampling/emd_cutoff": float(emd[challenging].min().item()),
        "eval/pace_sampling/challenging_weight_rank_mode": float(challenging_weight_mode == "rank"),
    }
    for group_id, group_name in enumerate(PACE_GROUP_NAMES):
        mask = group_ids == group_id
        count = int(mask.sum().item())
        stats[f"eval/pace_sampling/{group_name}/count"] = count
        stats[f"eval/pace_sampling/{group_name}/data_ratio"] = count / len(expected)
        stats[f"eval/pace_sampling/{group_name}/sampling_ratio"] = float(group_weights[group_id].item())
        stats[f"eval/pace_sampling/{group_name}/mean_emd"] = float(emd[mask].mean().item())
    if not torch.isfinite(probabilities).all() or not torch.isclose(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
    ):
        raise RuntimeError("PACE trajectory probabilities are not a finite normalized distribution")
    return {
        "probabilities": probabilities.to(torch.float32),
        "group_ids": group_ids,
        "group_weights": group_weights.to(torch.float32),
        "stats": stats,
    }


def _compute_global_emd_distribution(
    evaluation_metrics: dict,
    expected_motion_ids: list[int],
    *,
    clamp_min: float,
    clamp_max: float,
    scale: float,
) -> dict[str, tp.Any]:
    """Build one EMD-weighted distribution over every trajectory.

    This preserves the repository's original exponential prioritization rule,
    ``weight = 2 ** (scale * clamp(emd, clamp_min, clamp_max))``, but removes
    all high/low group quotas.
    """
    if not np.isfinite(clamp_min) or not np.isfinite(clamp_max) or not 0 <= clamp_min < clamp_max:
        raise ValueError("Global EMD clamps must satisfy 0 <= min < max")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Global EMD scale must be finite and positive")

    expected = [int(motion_id) for motion_id in expected_motion_ids]
    if not expected or expected != list(range(len(expected))):
        raise ValueError("Global EMD sampling requires expert motion ids in global order 0..N-1")
    emd_by_id: dict[int, float] = {}
    for motion_name, metric in evaluation_metrics.items():
        if "motion_id" not in metric or "emd" not in metric:
            raise ValueError(f"Tracking eval entry {motion_name!r} is missing motion_id or emd")
        motion_id = int(metric["motion_id"])
        if motion_id in emd_by_id:
            raise ValueError(f"Tracking eval returned duplicate motion_id={motion_id} ({motion_name!r})")
        emd = float(metric["emd"])
        if not np.isfinite(emd):
            raise ValueError(
                f"Tracking eval returned non-finite EMD for motion_id={motion_id}, motion={motion_name!r}: {emd}"
            )
        emd_by_id[motion_id] = emd
    expected_set = set(expected)
    if set(emd_by_id) != expected_set:
        missing = sorted(expected_set - set(emd_by_id))
        unexpected = sorted(set(emd_by_id) - expected_set)
        raise ValueError(f"Tracking eval motion ids do not match expert data: missing={missing}, unexpected={unexpected}")

    emd = torch.tensor([emd_by_id[motion_id] for motion_id in expected], dtype=torch.float64)
    logits = np.log(2.0) * float(scale) * torch.clamp(emd, min=clamp_min, max=clamp_max)
    probabilities = torch.softmax(logits, dim=0)
    if not torch.isfinite(probabilities).all() or not torch.isclose(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
    ):
        raise RuntimeError("Global EMD probabilities are not a finite normalized distribution")

    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum()
    effective_count = 1.0 / probabilities.square().sum()
    stats: dict[str, float | int] = {
        "eval/emd_sampling/global/count": len(expected),
        "eval/emd_sampling/global/data_ratio": 1.0,
        "eval/emd_sampling/global/sampling_ratio": 1.0,
        "eval/emd_sampling/global/mean_emd": float(emd.mean().item()),
        "eval/emd_sampling/global/min_emd": float(emd.min().item()),
        "eval/emd_sampling/global/max_emd": float(emd.max().item()),
        "eval/emd_sampling/global/min_probability": float(probabilities.min().item()),
        "eval/emd_sampling/global/max_probability": float(probabilities.max().item()),
        "eval/emd_sampling/global/entropy": float(entropy.item()),
        "eval/emd_sampling/global/effective_trajectory_count": float(effective_count.item()),
    }
    return {"probabilities": probabilities.to(torch.float32), "stats": stats}


def _log_emd_group_stats(
    stats: dict[str, float | int],
    *,
    timestep: int,
    use_wandb: bool,
    writer: SummaryWriter | None,
) -> None:
    if use_wandb:
        wandb.log(stats, step=timestep)
    if writer is not None:
        for key, value in stats.items():
            writer.add_scalar(key, value, timestep)
        writer.flush()


def _compute_b_convergence_metrics(
    previous_z: torch.Tensor,
    current_z: torch.Tensor,
    *,
    pairwise_size: int,
) -> dict[str, float]:
    """Compare the effective B embedding on an unchanged set of probe states."""
    previous_z = previous_z.detach().to(device="cpu", dtype=torch.float32)
    current_z = current_z.detach().to(device="cpu", dtype=torch.float32)
    if previous_z.ndim != 2 or previous_z.shape != current_z.shape:
        raise ValueError(
            "B convergence probes must be matching 2-D tensors, got "
            f"previous={tuple(previous_z.shape)} current={tuple(current_z.shape)}"
        )
    if previous_z.shape[0] < 2:
        raise ValueError("B convergence requires at least two probe states")
    if not torch.isfinite(previous_z).all() or not torch.isfinite(current_z).all():
        raise FloatingPointError("B convergence probe contains non-finite z values")

    eps = torch.finfo(torch.float32).eps
    previous_unit = previous_z / previous_z.norm(dim=-1, keepdim=True).clamp_min(eps)
    current_unit = current_z / current_z.norm(dim=-1, keepdim=True).clamp_min(eps)
    raw_cosine = (previous_unit * current_unit).sum(dim=-1).clamp(-1.0, 1.0)
    raw_angular = 2.0 * torch.asin(((previous_unit - current_unit).norm(dim=-1) * 0.5).clamp(0.0, 1.0))
    relative_l2 = (current_z - previous_z).norm(dim=-1) / previous_z.norm(dim=-1).clamp_min(eps)

    # FB embeddings may undergo a nearly global orthogonal rotation while their
    # pairwise geometry remains unchanged.  Procrustes alignment separates this
    # harmless coordinate motion from a real change in the representation.
    cross_covariance = current_unit.T @ previous_unit
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    aligned_current = current_unit @ (u @ vh)
    aligned_cosine = (previous_unit * aligned_current).sum(dim=-1).clamp(-1.0, 1.0)

    pair_count = min(int(pairwise_size), previous_z.shape[0])
    pair_indices = torch.linspace(0, previous_z.shape[0] - 1, steps=pair_count).round().to(torch.long)
    previous_pairwise = previous_unit[pair_indices] @ previous_unit[pair_indices].T
    current_pairwise = current_unit[pair_indices] @ current_unit[pair_indices].T
    off_diagonal = ~torch.eye(pair_count, dtype=torch.bool)
    pairwise_cosine_mae = (previous_pairwise - current_pairwise).abs()[off_diagonal].mean()

    return {
        "raw_cosine_mean": float(raw_cosine.mean().item()),
        "raw_cosine_p05": float(torch.quantile(raw_cosine, 0.05).item()),
        "raw_angular_degrees_mean": float(torch.rad2deg(raw_angular).mean().item()),
        "relative_l2_mean": float(relative_l2.mean().item()),
        "relative_l2_p95": float(torch.quantile(relative_l2, 0.95).item()),
        "aligned_cosine_mean": float(aligned_cosine.mean().item()),
        "aligned_cosine_p05": float(torch.quantile(aligned_cosine, 0.05).item()),
        "pairwise_cosine_mae": float(pairwise_cosine_mae.item()),
        "z_norm_mean": float(current_z.norm(dim=-1).mean().item()),
    }


def _log_grouped_tensorboard(
    writer: SummaryWriter | None,
    grouped_metrics: OrderedDict[str, list[tuple[str, float | int]]],
    timestep: int,
    *,
    steps_per_second: float,
    collection_time: float,
    learning_time: float,
    elapsed_time: float,
    eta_seconds: float,
    iteration: int,
) -> None:
    if writer is None:
        return
    for group_name, key, value in _iter_tracked_group_metrics(grouped_metrics):
        writer.add_scalar(_tensorboard_tag(group_name, key), value, timestep)
    writer.flush()


Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseMjlabTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig | TldrDistAuxAgentConfig


def _trajectory_output_keys(agent: Agent) -> list[str]:
    keys = [
        "observation",
        "action",
        "z",
        "terminated",
        "truncated",
        "step_count",
        "reward",
    ]
    if getattr(agent, "aux_rewards", None):
        keys.append("aux_rewards")
    if bool(getattr(getattr(agent, "model", None), "use_tracking_critic", False)):
        keys.extend(
            [
                "tracking_rewards",
                "expert_z_rollout",
                "reference_motion_id",
                "reference_step",
            ]
        )
    return keys


def _accumulate_metrics(
    total_metrics: dict[str, torch.Tensor] | None,
    metric_update_counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if total_metrics is None:
        total_metrics = {}
    for key, value in metrics.items():
        value = value.float()
        if key in total_metrics:
            total_metrics[key] = total_metrics[key] + value
        else:
            total_metrics[key] = value.clone()
        metric_update_counts[key] = metric_update_counts.get(key, 0) + 1
    return total_metrics, metric_update_counts


class PACEConfig(BaseConfig):
    """Stage-II Progress-Adaptive Capability Expansion configuration."""

    name: tp.Literal["PACEConfig"] = "PACEConfig"
    base_checkpoint: str
    challenging_fraction: float = 0.10
    acquisition_env_ratio: float = 0.80
    challenging_sampling_ratio: float = 0.80
    rollout_horizon: int = 24
    expert_rollout_max_steps: int = 500
    fixed_goal_steps: int = 100
    fixed_goal_probe_envs: int = 32
    fixed_goal_close_cosine: float = 0.95
    fixed_goal_plateau_window: int = 10
    fixed_goal_plateau_relative_change: float = 0.02
    fixed_goal_plateau_min_improvement: float = 0.20
    root_height_failure_threshold: float = 0.20
    lambda_base: float = 0.30
    lambda_gain: float = 5.0
    rho_reference: float = 0.60
    rho_ema_beta: float = 0.99
    lambda_max: float = 1.0
    grad_monitor_every_updates: int = 32
    context_encode_batch_size: int = 32768
    actor_lr_scale: float = 0.10
    actor_update_every: int = 4
    actor_warmup_windows: int = 10
    actor_lr_ramp_windows: int = 10
    gradient_scale_ema_beta: float = 0.95
    gradient_scale_min: float = 1.0
    gradient_scale_max: float = 100000.0
    gradient_scale_rate_limit: float = 2.0
    challenging_weight_mode: tp.Literal["absolute", "rank"] = "rank"
    retention_warn_ratio: float = 1.10
    retention_recovery_ratio: float = 1.15
    retention_stop_ratio: float = 1.25
    retention_recovery_evals: int = 2

    def model_post_init(self, context) -> None:
        del context
        for field_name in ("challenging_fraction", "acquisition_env_ratio", "challenging_sampling_ratio"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"PACE {field_name} must be finite and strictly between 0 and 1")
        for field_name in (
            "rollout_horizon",
            "expert_rollout_max_steps",
            "fixed_goal_steps",
            "fixed_goal_probe_envs",
            "fixed_goal_plateau_window",
            "grad_monitor_every_updates",
            "context_encode_batch_size",
            "actor_update_every",
            "actor_warmup_windows",
            "actor_lr_ramp_windows",
            "retention_recovery_evals",
        ):
            value = int(getattr(self, field_name))
            if field_name == "actor_warmup_windows":
                if value < 0:
                    raise ValueError(f"PACE {field_name} must be non-negative")
            elif value <= 0:
                raise ValueError(f"PACE {field_name} must be positive")
        if not np.isfinite(self.root_height_failure_threshold) or self.root_height_failure_threshold <= 0:
            raise ValueError("PACE root_height_failure_threshold must be finite and positive")
        if not 0 < self.fixed_goal_close_cosine <= 1:
            raise ValueError("PACE fixed_goal_close_cosine must lie in (0, 1]")
        if 2 * self.fixed_goal_plateau_window > self.fixed_goal_steps:
            raise ValueError("PACE fixed-goal plateau windows must fit inside fixed_goal_steps")
        if not 0 <= self.fixed_goal_plateau_relative_change < 1:
            raise ValueError("PACE fixed_goal_plateau_relative_change must lie in [0, 1)")
        if not 0 <= self.fixed_goal_plateau_min_improvement < 1:
            raise ValueError("PACE fixed_goal_plateau_min_improvement must lie in [0, 1)")
        if not 0 <= self.lambda_base <= self.lambda_max:
            raise ValueError("PACE lambda_base must lie in [0, lambda_max]")
        if self.lambda_gain < 0 or not np.isfinite(self.lambda_gain):
            raise ValueError("PACE lambda_gain must be finite and non-negative")
        if not 0 <= self.rho_reference <= 1:
            raise ValueError("PACE rho_reference must lie in [0, 1]")
        if not 0 <= self.rho_ema_beta < 1:
            raise ValueError("PACE rho_ema_beta must lie in [0, 1)")
        if not np.isfinite(self.actor_lr_scale) or not 0 < self.actor_lr_scale <= 1:
            raise ValueError("PACE actor_lr_scale must lie in (0, 1]")
        if not 0 <= self.gradient_scale_ema_beta < 1:
            raise ValueError("PACE gradient_scale_ema_beta must lie in [0, 1)")
        if not 0 < self.gradient_scale_min <= self.gradient_scale_max:
            raise ValueError("PACE gradient scale bounds must satisfy 0 < min <= max")
        if not np.isfinite(self.gradient_scale_rate_limit) or self.gradient_scale_rate_limit < 1:
            raise ValueError("PACE gradient_scale_rate_limit must be finite and at least one")
        if not 1 <= self.retention_warn_ratio <= self.retention_recovery_ratio <= self.retention_stop_ratio:
            raise ValueError(
                "PACE retention ratios must satisfy 1 <= warn <= recovery <= stop"
            )


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseMjlabConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    prioritization_high_group_ratio: float = 0.05
    prioritization_high_group_fraction: float | None = None
    pace: PACEConfig | None = None
    init_checkpoint: str | None = None
    track_b_convergence: bool = True
    b_convergence_probe_size: int = 512
    b_convergence_pairwise_size: int = 256
    b_convergence_encode_batch_size: int = 512
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    wandb_run_name: str | None = None

    # misc
    load_expert_data_from_motion_lib: bool = True
    buffer_device: str = "cpu"
    expert_buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    log_torso_contact_forces: bool = True
    torso_contact_force_threshold: float = 1.0
    distributed_rank: int = 0
    distributed_world_size: int = 1
    rank0_only_writes: bool = True
    checkpoint_rank_buffers: bool = True
    distributed_sync: bool = True
    distributed_global_steps: bool = True
    distributed_average_metrics: bool = True
    fail_on_nonfinite: bool = True
    nonfinite_check_model_every_updates: int = 0
    nonfinite_check_rollout_every_local_steps: int = 0

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    infra: dict = pydantic.Field(default_factory=dict)

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_expert_data_from_motion_lib and not isinstance(self.env, HumanoidVerseMjlabConfig):
            raise ValueError("Loading expert data from motion library is only supported for HumanoidVerseMjlabConfig")

        eval_configs = self.evaluations.values() if isinstance(self.evaluations, dict) else self.evaluations
        if self.prioritization:
            prioritization_evals = [
                eval_type
                for eval_type in eval_configs
                if isinstance(eval_type, HumanoidVerseMjlabTrackingEvaluationConfig)
            ]
            has_prioritization_eval = bool(prioritization_evals)
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")
            for eval_type in prioritization_evals:
                if eval_type.n_episodes_per_motion != 1:
                    raise ValueError("Prioritization requires n_episodes_per_motion=1")
                if eval_type.include_results_from_all_envs:
                    raise ValueError("Prioritization requires include_results_from_all_envs=False")
            if (
                not np.isfinite(self.prioritization_min_val)
                or not np.isfinite(self.prioritization_max_val)
                or not 0 <= self.prioritization_min_val < self.prioritization_max_val
            ):
                raise ValueError("Prioritization thresholds must satisfy 0 <= low < high")
            if not np.isfinite(self.prioritization_high_group_ratio) or not 0 < self.prioritization_high_group_ratio < 1:
                raise ValueError("prioritization_high_group_ratio must be finite and strictly between 0 and 1")
            if self.prioritization_high_group_fraction is not None and (
                not np.isfinite(self.prioritization_high_group_fraction)
                or not 0 < self.prioritization_high_group_fraction < 1
            ):
                raise ValueError(
                    "prioritization_high_group_fraction must be finite and strictly between 0 and 1"
                )

        if self.pace is not None:
            if not isinstance(self.agent, FBcprAuxAgentConfig):
                raise ValueError("PACE is currently supported only by the FBcprAuxAgent (the fb preset)")
            if not self.prioritization:
                raise ValueError("PACE requires full-dataset tracking eval and prioritization")
            if self.online_parallel_envs < 2:
                raise ValueError("PACE requires at least two environments per rank for fixed acquisition/consolidation roles")
            if isinstance(self.env.lafan_tail_path, list) and len(self.env.lafan_tail_path) != 1:
                raise ValueError("PACE requires one complete motion dataset, not a multi-source data mix")
        if self.init_checkpoint is not None and self.pace is not None:
            raise ValueError("init_checkpoint and PACE base_checkpoint are mutually exclusive")
        tracking_q = bool(
            isinstance(self.agent, FBcprAuxAgentConfig)
            and self.agent.model.use_tracking_critic
        )
        if tracking_q:
            if self.init_checkpoint is None and self.pace is None:
                raise ValueError("Tracking Q requires a model-only init_checkpoint or a PACE base checkpoint")
            if not self.use_trajectory_buffer:
                raise ValueError("Tracking Q requires use_trajectory_buffer=True")
            if self.online_parallel_envs < 2:
                raise ValueError("Tracking Q requires at least two environments for the exact 50/50 role split")
            if self.online_parallel_envs % 2:
                raise ValueError("Tracking Q requires an even number of environments for an exact 50/50 role split")
            if isinstance(self.env.lafan_tail_path, list) and len(self.env.lafan_tail_path) != 1:
                raise ValueError("Tracking Q currently requires one complete motion dataset")
            tracking_train = self.agent.train
            if not 0 < tracking_train.tracking_actor_grad_ratio <= 1:
                raise ValueError("tracking_actor_grad_ratio must lie in (0, 1]")
            if tracking_train.tracking_actor_warmup_updates < 0:
                raise ValueError("tracking_actor_warmup_updates must be non-negative")
            if tracking_train.tracking_actor_ramp_updates <= 0:
                raise ValueError("tracking_actor_ramp_updates must be positive")
            if tracking_train.tracking_grad_monitor_every_updates <= 0:
                raise ValueError("tracking_grad_monitor_every_updates must be positive")
            if not 0 <= tracking_train.tracking_grad_ema_beta < 1:
                raise ValueError("tracking_grad_ema_beta must lie in [0, 1)")
            if not 0 < tracking_train.tracking_grad_scale_min <= tracking_train.tracking_grad_scale_max:
                raise ValueError("tracking gradient scale bounds must satisfy 0 < min <= max")
            if tracking_train.tracking_grad_scale_rate_limit < 1:
                raise ValueError("tracking_grad_scale_rate_limit must be at least one")

        if self.track_b_convergence:
            if self.b_convergence_probe_size <= 0:
                raise ValueError("b_convergence_probe_size must be positive")
            if self.b_convergence_pairwise_size <= 1:
                raise ValueError("b_convergence_pairwise_size must be greater than one")
            if self.b_convergence_encode_batch_size <= 0:
                raise ValueError("b_convergence_encode_batch_size must be positive")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        return Workspace(self)


def _resolve_pace_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / CHECKPOINT_DIR_NAME / "model" / "model.safetensors").exists():
        checkpoint = checkpoint / CHECKPOINT_DIR_NAME
    required = (
        checkpoint / "model" / "model.safetensors",
        checkpoint / "config.json",
        checkpoint / "train_status.json",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"PACE base checkpoint is incomplete; missing: {missing}")
    return checkpoint


def _load_initial_model_weights(agent, checkpoint: Path, *, allow_tracking_missing: bool) -> None:
    model_path = checkpoint / "model" / "model.safetensors"
    missing, unexpected = safetensors.torch.load_model(
        agent._model,
        model_path,
        strict=False,
        device=str(agent.device),
    )
    allowed_prefixes = ("_tracking_critic.", "_target_tracking_critic.") if allow_tracking_missing else ()
    disallowed_missing = [name for name in missing if not name.startswith(allowed_prefixes)]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Initial model architecture does not match the current preset: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}, "
            f"allowed_new_tracking_parameters={sorted(set(missing) - set(disallowed_missing))}"
        )
    agent.apply_training_freezes()


def _load_pace_base_weights(agent, checkpoint: Path) -> None:
    _load_initial_model_weights(
        agent,
        checkpoint,
        allow_tracking_missing=bool(getattr(agent.cfg.model, "use_tracking_critic", False)),
    )


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    train_status_path = checkpoint_dir / "train_status.json"
    checkpoint_status = _initial_train_status(cfg)
    if train_status_path.exists():
        with train_status_path.open("r") as f:
            train_status = json.load(f)
        checkpoint_status = _normalize_train_status(train_status, cfg)

        print(
            f"Loading the agent at local_time={checkpoint_status['local_time']} "
            f"global_time={checkpoint_status['global_time']} optimizer_steps={checkpoint_status['optimizer_steps']}"
        )
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
        requested_tracking = bool(getattr(cfg.agent.model, "use_tracking_critic", False))
        checkpoint_tracking = bool(getattr(agent.cfg.model, "use_tracking_critic", False))
        if requested_tracking != checkpoint_tracking:
            raise RuntimeError(
                "Local checkpoint tracking-Q mode does not match the requested configuration: "
                f"checkpoint={checkpoint_tracking}, requested={requested_tracking}. "
                "Use a fresh work directory with --init-checkpoint for model-only initialization."
            )
        if requested_tracking:
            requested_ratio = float(cfg.agent.train.tracking_actor_grad_ratio)
            checkpoint_ratio = float(agent.cfg.train.tracking_actor_grad_ratio)
            if not np.isclose(requested_ratio, checkpoint_ratio):
                loaded_train = agent.cfg.train.model_copy(
                    update={"tracking_actor_grad_ratio": requested_ratio}
                )
                agent.cfg = agent.cfg.model_copy(update={"train": loaded_train})
                print(
                    "[Tracking Q] Applied runtime actor gradient ratio override after checkpoint load: "
                    f"checkpoint={checkpoint_ratio:.6g}, requested={requested_ratio:.6g}",
                    flush=True,
                )
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
        if cfg.pace is not None:
            base_checkpoint = _resolve_pace_checkpoint(cfg.pace.base_checkpoint)
            print(f"Loading PACE Stage-II model weights from {base_checkpoint}", flush=True)
            _load_pace_base_weights(agent, base_checkpoint)
        elif cfg.init_checkpoint is not None:
            initial_checkpoint = _resolve_pace_checkpoint(cfg.init_checkpoint)
            print(f"Loading model-only initialization weights from {initial_checkpoint}", flush=True)
            _load_initial_model_weights(
                agent,
                initial_checkpoint,
                allow_tracking_missing=bool(getattr(cfg.agent.model, "use_tracking_critic", False)),
            )
    return agent, cfg, checkpoint_status


def _global_step_scale(cfg: TrainConfig) -> int:
    if cfg.distributed_sync and cfg.distributed_global_steps and int(cfg.distributed_world_size) > 1:
        return int(cfg.distributed_world_size)
    return 1


def _effective_batch_size(cfg: TrainConfig) -> int:
    return int(cfg.agent.train.batch_size) * _global_step_scale(cfg)


def _num_envs_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.online_parallel_envs)


def _global_parallel_envs(cfg: TrainConfig) -> int:
    return _num_envs_per_rank(cfg) * _global_step_scale(cfg)


def _replay_capacity_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.buffer_size)


def _effective_replay_capacity(cfg: TrainConfig) -> int:
    return _replay_capacity_per_rank(cfg) * _global_step_scale(cfg)


def _trajectory_steps_per_rank(cfg: TrainConfig) -> int:
    if cfg.use_trajectory_buffer:
        return int(cfg.buffer_size) // max(_num_envs_per_rank(cfg), 1)
    return int(cfg.buffer_size)


def _tensor_nonfinite_summary(value: tp.Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.is_floating_point(value):
            return None
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        bad_count = int(bad.sum().item())
        with torch.no_grad():
            finite_values = value[finite]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else None
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else None
            bad_indices = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
            bad_values = value[bad][:5].detach().cpu().tolist()
        return (
            f"type=torch shape={tuple(value.shape)} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            return None
        finite = np.isfinite(value)
        if bool(finite.all()):
            return None
        bad = ~finite
        bad_count = int(bad.sum())
        finite_values = value[finite]
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None
        bad_indices = np.argwhere(bad)[:5].tolist()
        bad_values = value[bad][:5].tolist()
        return (
            f"type=numpy shape={value.shape} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return None
        return f"type=scalar value={value}"
    return None


def _iter_nonfinite(value: tp.Any, prefix: str = "") -> tp.Iterator[tuple[str, str]]:
    summary = _tensor_nonfinite_summary(value)
    if summary is not None:
        yield prefix or "<root>", summary
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nonfinite(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_nonfinite(item, child_prefix)


def _assert_finite(value: tp.Any, *, label: str, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    bad = list(_iter_nonfinite(value, label))
    if not bad:
        return
    details = "\n".join(f"  - {path}: {summary}" for path, summary in bad[:20])
    more = "" if len(bad) <= 20 else f"\n  ... {len(bad) - 20} more non-finite fields"
    raise FloatingPointError(
        "Non-finite value detected "
        f"(rank={rank}, local_time={local_time}, global_time={global_time}, optimizer_steps={optimizer_steps}):\n"
        f"{details}{more}"
    )


def _assert_model_finite(module: torch.nn.Module, *, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    for name, param in module.named_parameters():
        _assert_finite(
            param,
            label=f"model.parameters.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )
    for name, buffer in module.named_buffers():
        _assert_finite(
            buffer,
            label=f"model.buffers.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )


def _distributed_loss_mode(cfg: TrainConfig) -> str:
    if cfg.distributed_sync and int(cfg.distributed_world_size) > 1:
        return "local_loss_average"
    return "single_rank"


def _initial_train_status(cfg: TrainConfig) -> dict[str, tp.Any]:
    status = {
        "local_time": 0,
        "global_time": 0,
        "optimizer_steps": 0,
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }
    if cfg.pace is not None:
        status["pace"] = {
            "latest_rho": float(cfg.pace.rho_reference),
            "rho_ema": float(cfg.pace.rho_reference),
            "lambda_con": float(cfg.pace.lambda_base),
            "windows_completed": 0,
            "fixed_group_ids": None,
            "baseline_emd": None,
            "best_mastered_emd": None,
            "retention_violation_count": 0,
            "retention_recovery": False,
            "retention_warning": False,
            "actor_runtime": {},
        }
    if bool(getattr(cfg.agent.model, "use_tracking_critic", False)):
        status["tracking_q"] = {}
    return status


def _normalize_train_status(train_status: dict[str, tp.Any], cfg: TrainConfig) -> dict[str, tp.Any]:
    status = _initial_train_status(cfg)
    scale = _global_step_scale(cfg)
    current_world_size = int(cfg.distributed_world_size)
    if "world_size" not in train_status and current_world_size > 1:
        raise RuntimeError(
            "Cannot safely resume this checkpoint because checkpoint/train_status.json does not record world_size. "
            "Distributed checkpoints contain rank-local replay buffer shards; start a fresh work_dir or migrate buffers explicitly."
        )
    checkpoint_world_size = int(train_status.get("world_size", current_world_size))
    if checkpoint_world_size != current_world_size:
        raise RuntimeError(
            "Cannot resume checkpoint with a different distributed world_size: "
            f"checkpoint world_size={checkpoint_world_size}, current world_size={current_world_size}. "
            "Rank-local replay buffer shards cannot be automatically migrated; use a matching GPU count, "
            "start a fresh work_dir, or perform an explicit buffer migration."
        )
    if "local_time" in train_status:
        status["local_time"] = int(train_status["local_time"])
        status["global_time"] = int(train_status.get("global_time", status["local_time"] * scale))
    else:
        legacy_time = int(train_status.get("time", 0))
        status["global_time"] = legacy_time
        status["local_time"] = (legacy_time + scale - 1) // scale
    status["optimizer_steps"] = int(train_status.get("optimizer_steps", 0))
    status["world_size"] = checkpoint_world_size
    status["loss_mode"] = str(train_status.get("loss_mode", _distributed_loss_mode(cfg)))
    status["effective_batch_size"] = int(train_status.get("effective_batch_size", _effective_batch_size(cfg)))
    if cfg.pace is not None:
        loaded_pace = train_status.get("pace", {})
        status["pace"] = {
            "latest_rho": float(loaded_pace.get("latest_rho", cfg.pace.rho_reference)),
            "rho_ema": float(loaded_pace.get("rho_ema", cfg.pace.rho_reference)),
            "lambda_con": float(loaded_pace.get("lambda_con", cfg.pace.lambda_base)),
            "windows_completed": int(loaded_pace.get("windows_completed", 0)),
            "fixed_group_ids": loaded_pace.get("fixed_group_ids"),
            "baseline_emd": loaded_pace.get("baseline_emd"),
            "best_mastered_emd": loaded_pace.get("best_mastered_emd"),
            "retention_violation_count": int(loaded_pace.get("retention_violation_count", 0)),
            "retention_recovery": bool(loaded_pace.get("retention_recovery", False)),
            "retention_warning": bool(loaded_pace.get("retention_warning", False)),
            "actor_runtime": dict(loaded_pace.get("actor_runtime", {})),
        }
    if bool(getattr(cfg.agent.model, "use_tracking_critic", False)):
        status["tracking_q"] = dict(train_status.get("tracking_q", {}))
    return status


def _make_train_status(
    cfg: TrainConfig,
    *,
    local_time: int,
    global_time: int,
    optimizer_steps: int,
    pace_state: dict[str, tp.Any] | None = None,
    tracking_q_state: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    status = {
        "time": int(global_time),
        "local_time": int(local_time),
        "global_time": int(global_time),
        "optimizer_steps": int(optimizer_steps),
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }
    if cfg.pace is not None:
        status["pace"] = dict(pace_state or _initial_train_status(cfg)["pace"])
    if bool(getattr(cfg.agent.model, "use_tracking_critic", False)):
        status["tracking_q"] = dict(tracking_q_state or {})
    return status


def init_wandb(cfg: TrainConfig):
    from pathlib import Path
    exp_name = cfg.wandb_run_name if cfg.wandb_run_name else Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb_dir = Path(cfg.work_dir) / "_wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        entity=cfg.wandb_ename,
        project=cfg.wandb_pname,
        group=cfg.wandb_gname,
        name=exp_name,
        config=wandb_config,
        dir=str(wandb_dir),
    )


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.distributed_rank = int(self.cfg.distributed_rank)
        self.distributed_world_size = int(self.cfg.distributed_world_size)
        self._write_shared_artifacts = (not self.cfg.rank0_only_writes) or self.distributed_rank == 0

        # MJLab environments are created once and shared with evaluation.
        if isinstance(cfg.env, HumanoidVerseMjlabConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if self._write_shared_artifacts and isinstance(cfg.env, HumanoidVerseMjlabConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self._write_shared_artifacts else None
        self.tb_writer = SummaryWriter(log_dir=str(self.work_dir / "tensorboard")) if self.distributed_rank == 0 else None

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_status = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self._checkpoint_local_time = int(self._checkpoint_status["local_time"])
        self._checkpoint_global_time = int(self._checkpoint_status["global_time"])
        self._optimizer_steps = int(self._checkpoint_status["optimizer_steps"])
        self._pace_roles: np.ndarray | None = None
        self._tracking_roles: np.ndarray | None = None
        self._pace_alive: np.ndarray | None = None
        self._pace_window_step = 0
        self._pace_valid_acquisition = 0
        self._pace_valid_consolidation = 0
        self._pace_total_acquisition = 0
        self._pace_total_consolidation = 0
        self._pace_fixed_group_ids: torch.Tensor | None = None
        self._pace_baseline_emd: dict[str, float] | None = None
        self._pace_best_mastered_emd: float | None = None
        self._pace_retention_violation_count = 0
        self._pace_retention_recovery = False
        self._pace_retention_warning = False
        self._pace_retention_stop_requested = False
        self._fixed_goal_probe_env_ids: np.ndarray | None = None
        self._fixed_goal_probe_active: np.ndarray | None = None
        self._fixed_goal_probe_history: np.ndarray | None = None
        self._fixed_goal_probe_rmse_history: np.ndarray | None = None
        self._fixed_goal_probe_close_steps: np.ndarray | None = None
        self._fixed_goal_probe_plateau_steps: np.ndarray | None = None
        self._fixed_goal_probe_totals = self._new_fixed_goal_probe_totals()
        self._pace_latest_rho = float(self.cfg.pace.rho_reference) if self.cfg.pace is not None else 0.0
        if self.cfg.pace is not None:
            pace_status = self._checkpoint_status["pace"]
            self._pace_latest_rho = float(pace_status["latest_rho"])
            self._pace_rho_ema = float(pace_status["rho_ema"])
            self._pace_lambda_con = float(pace_status["lambda_con"])
            self._pace_windows_completed = int(pace_status["windows_completed"])
            fixed_group_ids = pace_status.get("fixed_group_ids")
            if fixed_group_ids is not None:
                self._pace_fixed_group_ids = torch.tensor(fixed_group_ids, dtype=torch.long)
            baseline_emd = pace_status.get("baseline_emd")
            if baseline_emd is not None:
                self._pace_baseline_emd = {str(key): float(value) for key, value in baseline_emd.items()}
            best_mastered_emd = pace_status.get("best_mastered_emd")
            if best_mastered_emd is not None:
                self._pace_best_mastered_emd = float(best_mastered_emd)
            self._pace_retention_violation_count = int(pace_status.get("retention_violation_count", 0))
            self._pace_retention_recovery = bool(pace_status.get("retention_recovery", False))
            self._pace_retention_warning = bool(pace_status.get("retention_warning", False))
            reference_path = self.work_dir / CHECKPOINT_DIR_NAME / "pace_reference_actor.safetensors"
            if self._checkpoint_local_time > 0 and not reference_path.exists():
                raise FileNotFoundError(
                    "Cannot resume PACE without checkpoint/pace_reference_actor.safetensors; "
                    "the immutable Stage-I policy reference would be lost"
                )
            self.agent.enable_pace(
                lambda_con=self._pace_lambda_con,
                fixed_goal_steps=self.cfg.pace.fixed_goal_steps,
                expert_rollout_max_steps=self.cfg.pace.expert_rollout_max_steps,
                grad_monitor_every_updates=self.cfg.pace.grad_monitor_every_updates,
                context_encode_batch_size=self.cfg.pace.context_encode_batch_size,
                actor_update_every=self.cfg.pace.actor_update_every,
                gradient_scale_ema_beta=self.cfg.pace.gradient_scale_ema_beta,
                gradient_scale_min=self.cfg.pace.gradient_scale_min,
                gradient_scale_max=self.cfg.pace.gradient_scale_max,
                gradient_scale_rate_limit=self.cfg.pace.gradient_scale_rate_limit,
                reference_actor_path=reference_path if reference_path.exists() else None,
            )
            self.agent.load_pace_checkpoint_state(pace_status.get("actor_runtime"))
        else:
            self._pace_rho_ema = 0.0
            self._pace_lambda_con = 0.0
            self._pace_windows_completed = 0
        if bool(getattr(self.cfg.agent.model, "use_tracking_critic", False)):
            self.agent.load_tracking_checkpoint_state(self._checkpoint_status.get("tracking_q"))
        if self.cfg.distributed_sync:
            broadcast_agent_state(self.agent, src=0)
        if hasattr(self.agent, "set_train_mode"):
            self.agent.set_train_mode(True)
        else:
            self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {
            name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()
        } if self._write_shared_artifacts else {}

        if self._write_shared_artifacts and self.cfg.use_wandb:
            init_wandb(self.cfg)

        if self._write_shared_artifacts:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseMjlabTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None
        self._b_probe_indices: torch.Tensor | None = None
        self._b_previous_z: torch.Tensor | None = None
        self._b_previous_timestep: int | None = None
        self._b_probe_restore_attempted = False

    def _restore_b_convergence_probe(self, *, total_expert_states: int) -> None:
        if self._b_probe_restore_attempted:
            return
        self._b_probe_restore_attempted = True
        probe_path = self.work_dir / B_CONVERGENCE_PROBE_FILENAME
        if not probe_path.exists():
            return
        try:
            payload = torch.load(probe_path, map_location="cpu", weights_only=True)
            saved_timestep = int(payload["timestep"].item())
            saved_total = int(payload["total_expert_states"].item())
            indices = payload["indices"].to(dtype=torch.long, device="cpu")
            previous_z = payload["z"].to(dtype=torch.float32, device="cpu")
            valid = (
                saved_timestep <= self._checkpoint_global_time
                and saved_total == total_expert_states
                and indices.ndim == 1
                and len(indices) == min(self.cfg.b_convergence_probe_size, total_expert_states)
                and bool((indices >= 0).all().item())
                and bool((indices < total_expert_states).all().item())
                and previous_z.ndim == 2
                and previous_z.shape[0] == len(indices)
                and bool(torch.isfinite(previous_z).all().item())
            )
            if not valid:
                print(
                    "[B convergence] Ignoring an incompatible or newer saved probe baseline; "
                    "the current eval will establish a new baseline.",
                    flush=True,
                )
                return
            self._b_probe_indices = indices
            self._b_previous_z = previous_z
            self._b_previous_timestep = saved_timestep
            print(
                f"[B convergence] Restored {len(indices)} probe embeddings from timestep {saved_timestep}.",
                flush=True,
            )
        except Exception as exc:
            print(f"[B convergence] Could not restore probe baseline: {exc}. Starting a new baseline.", flush=True)

    def _encode_b_convergence_probe(self, expert_buffer) -> torch.Tensor | None:
        model = self.agent._model
        if not hasattr(model, "_backward_map") or not hasattr(model, "backward_map") or not hasattr(model, "project_z"):
            return None
        observation_storage = expert_buffer.storage.get("observation")
        if not isinstance(observation_storage, dict) or not observation_storage:
            raise RuntimeError("B convergence requires expert_buffer.storage['observation']")
        tensor_leaves = [value for value in observation_storage.values() if isinstance(value, torch.Tensor)]
        if not tensor_leaves:
            raise RuntimeError("B convergence expert observations contain no tensor leaves")
        total_expert_states = int(tensor_leaves[0].shape[0])
        if any(int(value.shape[0]) != total_expert_states for value in tensor_leaves):
            raise RuntimeError("B convergence expert observation leaves have inconsistent lengths")
        if total_expert_states < 2:
            print("[B convergence] Skipping: fewer than two expert states are available.", flush=True)
            return None

        self._restore_b_convergence_probe(total_expert_states=total_expert_states)
        if self._b_probe_indices is None:
            probe_count = min(int(self.cfg.b_convergence_probe_size), total_expert_states)
            self._b_probe_indices = (
                torch.linspace(0, total_expert_states - 1, steps=probe_count).round().to(dtype=torch.long, device="cpu")
            )

        try:
            model_device = next(model._backward_map.parameters()).device
        except StopIteration:
            model_device = torch.device(self.cfg.agent.model.device)
        encoded_batches = []
        with torch.no_grad():
            for index_batch in self._b_probe_indices.split(int(self.cfg.b_convergence_encode_batch_size)):
                observations = tree_map(
                    lambda value: value.index_select(0, index_batch.to(value.device)).to(model_device),
                    observation_storage,
                )
                z = model.project_z(model.backward_map(observations))
                encoded_batches.append(z.detach().to(device="cpu", dtype=torch.float32))
        return torch.cat(encoded_batches, dim=0)

    def _save_b_convergence_probe(self, *, timestep: int, total_expert_states: int, z: torch.Tensor) -> None:
        if not self._write_shared_artifacts or self._b_probe_indices is None:
            return
        probe_path = self.work_dir / B_CONVERGENCE_PROBE_FILENAME
        temporary_path = probe_path.with_suffix(probe_path.suffix + ".tmp")
        torch.save(
            {
                "version": torch.tensor(1, dtype=torch.int64),
                "timestep": torch.tensor(int(timestep), dtype=torch.int64),
                "total_expert_states": torch.tensor(int(total_expert_states), dtype=torch.int64),
                "indices": self._b_probe_indices.to(dtype=torch.int64, device="cpu"),
                "z": z.detach().to(dtype=torch.float32, device="cpu"),
            },
            temporary_path,
        )
        os.replace(temporary_path, probe_path)

    def _log_b_convergence(self, *, timestep: int, replay_buffer: dict[str, tp.Any]) -> None:
        if not self.cfg.track_b_convergence or not self._write_shared_artifacts:
            return
        expert_buffer = replay_buffer.get("expert_slicer")
        if expert_buffer is None:
            print("[B convergence] Skipping: no expert buffer is available.", flush=True)
            return
        current_z = self._encode_b_convergence_probe(expert_buffer)
        if current_z is None or self._b_probe_indices is None:
            return

        total_expert_states = int(next(iter(expert_buffer.storage["observation"].values())).shape[0])
        metrics: dict[str, float | int] = {
            "eval/b_convergence/has_previous": int(self._b_previous_z is not None),
            "eval/b_convergence/probe_count": int(current_z.shape[0]),
            "eval/b_convergence/z_norm_mean": float(current_z.norm(dim=-1).mean().item()),
        }
        if self._b_previous_z is not None and self._b_previous_timestep is not None:
            drift = _compute_b_convergence_metrics(
                self._b_previous_z,
                current_z,
                pairwise_size=self.cfg.b_convergence_pairwise_size,
            )
            metrics.update({f"eval/b_convergence/{key}": value for key, value in drift.items()})
            metrics["eval/b_convergence/previous_timestep"] = int(self._b_previous_timestep)
            metrics["eval/b_convergence/eval_gap_steps"] = int(timestep - self._b_previous_timestep)
            print(
                "[B convergence] "
                f"step={timestep} gap={timestep - self._b_previous_timestep} probes={len(current_z)} "
                f"raw_cos={drift['raw_cosine_mean']:.6f} aligned_cos={drift['aligned_cosine_mean']:.6f} "
                f"relative_l2={drift['relative_l2_mean']:.6f} pairwise_mae={drift['pairwise_cosine_mae']:.6f}",
                flush=True,
            )
        else:
            print(f"[B convergence] Established a fixed baseline with {len(current_z)} expert states at step={timestep}.", flush=True)

        if self.cfg.use_wandb:
            wandb.log(metrics, step=timestep)
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, timestep)
            self.tb_writer.flush()

        self._b_previous_z = current_z
        self._b_previous_timestep = int(timestep)
        self._save_b_convergence_probe(timestep=timestep, total_expert_states=total_expert_states, z=current_z)

    def _checkpoint_buffer_path(self, checkpoint_dir: Path) -> Path:
        if self.cfg.checkpoint_rank_buffers and self.distributed_world_size > 1:
            return checkpoint_dir / "buffers" / f"train_rank_{self.distributed_rank}"
        return checkpoint_dir / "buffers" / "train"

    def train(self):
        self.start_time = time.time()
        try:
            self.train_online()
        finally:
            if self.tb_writer is not None:
                self.tb_writer.close()

    def _get_torso_contact_force_metrics(self, train_env) -> dict[str, float]:
        if not self.cfg.log_torso_contact_forces:
            return {}

        raw_env = getattr(train_env, "_env", train_env)
        simulator = getattr(raw_env, "simulator", None)
        torso_index = getattr(raw_env, "torso_index", None)
        if simulator is None or torso_index is None:
            return {}

        with torch.no_grad():
            torso_force = simulator.contact_forces[:, torso_index, :].float()
            torso_norm = torch.linalg.vector_norm(torso_force, dim=-1)
            torso_mean = torso_force.mean(dim=0)
            torso_force_z = torso_force[:, 2]
            torso_z = simulator._rigid_body_pos[:, torso_index, 2].float()
            contact_mask = torso_norm > self.cfg.torso_contact_force_threshold
            contact_count = contact_mask.sum()
            num_envs = max(torso_norm.numel(), 1)

            metrics = {
                "torso_contact_force_mean": np.round(torso_norm.mean().item(), 6),
                "torso_contact_force_max": np.round(torso_norm.max().item(), 6),
                "torso_contact_force_p95": np.round(torch.quantile(torso_norm, 0.95).item(), 6),
                "torso_contact_force_contact_count": int(contact_count.item()),
                "torso_contact_force_contact_frac": np.round((contact_count.float() / num_envs).item(), 6),
                "torso_contact_force_mean_x": np.round(torso_mean[0].item(), 6),
                "torso_contact_force_mean_y": np.round(torso_mean[1].item(), 6),
                "torso_contact_force_mean_z": np.round(torso_mean[2].item(), 6),
                "torso_contact_force_z_mean": np.round(torso_force_z.mean().item(), 6),
                "torso_contact_force_z_max": np.round(torso_force_z.max().item(), 6),
                "torso_contact_force_z_p95": np.round(torch.quantile(torso_force_z, 0.95).item(), 6),
                "torso_z_mean": np.round(torso_z.mean().item(), 6),
                "torso_z_min": np.round(torso_z.min().item(), 6),
                "torso_z_p05": np.round(torch.quantile(torso_z, 0.05).item(), 6),
            }
            if contact_count.item() > 0:
                torso_z_contact = torso_z[contact_mask]
                metrics.update(
                    {
                        "torso_z_contact_mean": np.round(torso_z_contact.mean().item(), 6),
                        "torso_z_contact_min": np.round(torso_z_contact.min().item(), 6),
                        "torso_z_contact_p05": np.round(torch.quantile(torso_z_contact, 0.05).item(), 6),
                    }
                )
            else:
                metrics.update(
                    {
                        "torso_z_contact_mean": 0.0,
                        "torso_z_contact_min": 0.0,
                        "torso_z_contact_p05": 0.0,
                    }
                )
            return metrics

    def _initialize_pace_runtime(self, train_env, expert_buffer=None) -> None:
        if self.cfg.pace is None:
            return
        if expert_buffer is None:
            raise RuntimeError("PACE initialization requires the loaded expert trajectory buffer")
        motion_ids = [int(motion_id) for motion_id in expert_buffer.motion_ids]
        if motion_ids != list(range(len(motion_ids))):
            raise ValueError("PACE expert trajectories must be stored in global motion-id order 0..N-1")
        expert_motion_lengths = torch.tensor(
            [expert_buffer.motion_length(motion_id) for motion_id in motion_ids],
            dtype=torch.long,
        )
        num_envs = int(self.cfg.online_parallel_envs)
        num_acquisition = min(max(int(round(num_envs * self.cfg.pace.acquisition_env_ratio)), 1), num_envs - 1)
        rng = np.random.default_rng(int(self.cfg.seed))
        permutation = rng.permutation(num_envs)
        roles = np.zeros(num_envs, dtype=bool)
        roles[permutation[:num_acquisition]] = True
        self._pace_roles = roles
        self._pace_alive = np.ones(num_envs, dtype=bool)
        train_env._env.configure_pace(
            roles,
            expert_motion_lengths=expert_motion_lengths,
            seq_length=int(self.cfg.agent.model.seq_length),
            expert_rollout_max_steps=self.cfg.pace.expert_rollout_max_steps,
            root_height_failure_threshold=self.cfg.pace.root_height_failure_threshold,
        )
        self._apply_pace_actor_schedule()
        self._update_agent_pace_metrics(
            valid_acquisition=0,
            valid_consolidation=0,
            total_acquisition=0,
            total_consolidation=0,
        )
        self._initialize_fixed_goal_probe(roles)
        if self._write_shared_artifacts:
            print(
                "[PACE] Fixed environment roles: "
                f"acquisition={int(roles.sum())}, consolidation={int((~roles).sum())}, "
                f"horizon={self.cfg.pace.rollout_horizon}",
                flush=True,
            )

    @staticmethod
    def _new_fixed_goal_probe_totals() -> dict[str, float]:
        return {
            "completed": 0.0,
            "interrupted": 0.0,
            "close": 0.0,
            "plateau": 0.0,
            "arrival": 0.0,
            "close_step_sum": 0.0,
            "plateau_step_sum": 0.0,
            "arrival_step_sum": 0.0,
            "arrival_step_capped_sum": 0.0,
            "initial_cosine_sum": 0.0,
            "final_cosine_sum": 0.0,
            "best_cosine_sum": 0.0,
            "initial_rmse_sum": 0.0,
            "final_rmse_sum": 0.0,
            "best_rmse_sum": 0.0,
        }

    def _initialize_fixed_goal_probe(self, pace_roles: np.ndarray) -> None:
        if self.cfg.pace is None:
            return
        candidates = np.flatnonzero(~pace_roles)
        count = min(int(self.cfg.pace.fixed_goal_probe_envs), len(candidates))
        if count <= 0:
            raise RuntimeError("PACE fixed-goal probe requires at least one consolidation environment")
        rng = np.random.default_rng(int(self.cfg.seed) + 7919)
        self._fixed_goal_probe_env_ids = np.sort(rng.choice(candidates, size=count, replace=False))
        horizon = int(self.cfg.pace.fixed_goal_steps)
        self._fixed_goal_probe_active = np.zeros(count, dtype=bool)
        self._fixed_goal_probe_history = np.full((count, horizon), np.nan, dtype=np.float32)
        self._fixed_goal_probe_rmse_history = np.full((count, horizon), np.nan, dtype=np.float32)
        self._fixed_goal_probe_close_steps = np.full(count, -1, dtype=np.int32)
        self._fixed_goal_probe_plateau_steps = np.full(count, -1, dtype=np.int32)
        self._fixed_goal_probe_totals = self._new_fixed_goal_probe_totals()
        if self._write_shared_artifacts:
            print(
                "[PACE] Fixed-goal convergence probe: "
                f"envs_per_rank={count}, horizon={horizon}, "
                f"close_cosine={self.cfg.pace.fixed_goal_close_cosine}, "
                f"plateau_window={self.cfg.pace.fixed_goal_plateau_window}",
                flush=True,
            )

    @torch.no_grad()
    def _record_fixed_goal_progress(
        self,
        *,
        next_observation: dict[str, tp.Any],
        target_z: torch.Tensor,
        step_count: torch.Tensor,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        if self.cfg.pace is None or self._fixed_goal_probe_env_ids is None:
            return
        env_ids = self._fixed_goal_probe_env_ids
        active = self._fixed_goal_probe_active
        history = self._fixed_goal_probe_history
        rmse_history = self._fixed_goal_probe_rmse_history
        close_steps = self._fixed_goal_probe_close_steps
        plateau_steps = self._fixed_goal_probe_plateau_steps
        if active is None or history is None or rmse_history is None or close_steps is None or plateau_steps is None:
            raise RuntimeError("PACE fixed-goal probe was not fully initialized")

        horizon = int(self.cfg.pace.fixed_goal_steps)
        steps = step_count.reshape(-1).detach().cpu().numpy().astype(np.int64)[env_ids]
        cycle_steps = steps % horizon
        done = np.logical_or(np.asarray(terminated).reshape(-1), np.asarray(truncated).reshape(-1))[env_ids]
        starting = cycle_steps == 0
        for row in np.flatnonzero(starting):
            if active[row]:
                self._fixed_goal_probe_totals["interrupted"] += 1.0
            active[row] = True
            history[row].fill(np.nan)
            rmse_history[row].fill(np.nan)
            close_steps[row] = -1
            plateau_steps[row] = -1

        for row in np.flatnonzero(done & active):
            self._fixed_goal_probe_totals["interrupted"] += 1.0
            active[row] = False

        valid_rows = np.flatnonzero(active & ~done)
        if len(valid_rows) == 0:
            return
        valid_env_ids = env_ids[valid_rows]
        source = {key: value for key, value in next_observation.items() if key not in {"time", "history"}}

        def select_to_device(value):
            if torch.is_tensor(value):
                indices = torch.as_tensor(valid_env_ids, device=value.device, dtype=torch.long)
                return value.index_select(0, indices).to(self.agent.device)
            array = np.asarray(value)
            return torch.as_tensor(
                array[valid_env_ids],
                dtype=dtype_numpytotorch_lower_precision(array.dtype),
                device=self.agent.device,
            )

        probe_observation = tree_map(select_to_device, source)
        current_z = self.agent._model.goal_inference(probe_observation)
        target = target_z.index_select(
            0, torch.as_tensor(valid_env_ids, device=target_z.device, dtype=torch.long)
        )
        cosine = torch.nn.functional.cosine_similarity(current_z.float(), target.float(), dim=-1).clamp(-1.0, 1.0)
        latent_rmse = torch.sqrt(torch.mean((current_z.float() - target.float()).square(), dim=-1))
        cosine_values = cosine.detach().cpu().numpy()
        rmse_values = latent_rmse.detach().cpu().numpy()
        window = int(self.cfg.pace.fixed_goal_plateau_window)

        for offset, row in enumerate(valid_rows):
            elapsed = int(cycle_steps[row]) + 1
            cosine_value = float(cosine_values[offset])
            cosine_error = 1.0 - cosine_value
            rmse_value = float(rmse_values[offset])
            history[row, elapsed - 1] = cosine_error
            rmse_history[row, elapsed - 1] = rmse_value
            if close_steps[row] < 0 and cosine_value >= float(self.cfg.pace.fixed_goal_close_cosine):
                close_steps[row] = elapsed
            if plateau_steps[row] < 0 and elapsed >= 2 * window:
                previous = float(np.mean(history[row, elapsed - 2 * window : elapsed - window]))
                recent = float(np.mean(history[row, elapsed - window : elapsed]))
                initial = float(history[row, 0])
                relative_change = abs(recent - previous) / max(previous, 1.0e-6)
                improvement = (initial - recent) / max(initial, 1.0e-6)
                if (
                    relative_change <= float(self.cfg.pace.fixed_goal_plateau_relative_change)
                    and improvement >= float(self.cfg.pace.fixed_goal_plateau_min_improvement)
                ):
                    plateau_steps[row] = elapsed
            if elapsed != horizon:
                continue

            close_step = int(close_steps[row])
            plateau_step = int(plateau_steps[row])
            candidates = [value for value in (close_step, plateau_step) if value > 0]
            arrival_step = min(candidates) if candidates else horizon
            totals = self._fixed_goal_probe_totals
            totals["completed"] += 1.0
            totals["close"] += float(close_step > 0)
            totals["plateau"] += float(plateau_step > 0)
            totals["arrival"] += float(bool(candidates))
            totals["close_step_sum"] += float(max(close_step, 0))
            totals["plateau_step_sum"] += float(max(plateau_step, 0))
            totals["arrival_step_sum"] += float(arrival_step if candidates else 0)
            totals["arrival_step_capped_sum"] += float(arrival_step)
            totals["initial_cosine_sum"] += 1.0 - float(history[row, 0])
            totals["final_cosine_sum"] += cosine_value
            totals["best_cosine_sum"] += 1.0 - float(np.nanmin(history[row]))
            totals["initial_rmse_sum"] += float(rmse_history[row, 0])
            totals["final_rmse_sum"] += rmse_value
            totals["best_rmse_sum"] += float(np.nanmin(rmse_history[row]))
            active[row] = False

    def _consume_fixed_goal_probe_metrics(self) -> dict[str, float]:
        if self.cfg.pace is None or self._fixed_goal_probe_env_ids is None:
            return {}
        keys = tuple(self._fixed_goal_probe_totals)
        totals = torch.tensor(
            [self._fixed_goal_probe_totals[key] for key in keys],
            dtype=torch.float64,
            device=self.agent.device,
        )
        if self.cfg.distributed_sync:
            import torch.distributed as dist

            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        values = dict(zip(keys, totals.detach().cpu().tolist(), strict=True))
        self._fixed_goal_probe_totals = self._new_fixed_goal_probe_totals()
        completed = values["completed"]
        metrics = {
            "goal_reach/probe_envs_per_rank": float(len(self._fixed_goal_probe_env_ids)),
            "goal_reach/completed_goals": completed,
            "goal_reach/interrupted_goals": values["interrupted"],
        }
        if completed <= 0:
            return metrics
        metrics.update(
            {
                "goal_reach/close_rate": values["close"] / completed,
                "goal_reach/plateau_rate": values["plateau"] / completed,
                "goal_reach/arrival_rate": values["arrival"] / completed,
                "goal_reach/close_step_mean": values["close_step_sum"] / max(values["close"], 1.0),
                "goal_reach/plateau_step_mean": values["plateau_step_sum"] / max(values["plateau"], 1.0),
                "goal_reach/arrival_step_mean": values["arrival_step_sum"] / max(values["arrival"], 1.0),
                "goal_reach/arrival_step_capped_mean": values["arrival_step_capped_sum"] / completed,
                "goal_reach/z_cosine_initial": values["initial_cosine_sum"] / completed,
                "goal_reach/z_cosine_final": values["final_cosine_sum"] / completed,
                "goal_reach/z_cosine_best": values["best_cosine_sum"] / completed,
                "goal_reach/z_rmse_initial": values["initial_rmse_sum"] / completed,
                "goal_reach/z_rmse_final": values["final_rmse_sum"] / completed,
                "goal_reach/z_rmse_best": values["best_rmse_sum"] / completed,
            }
        )
        return metrics

    def _initialize_tracking_runtime(self, train_env, expert_buffer=None) -> None:
        if not bool(getattr(self.cfg.agent.model, "use_tracking_critic", False)):
            return
        if expert_buffer is None:
            raise RuntimeError("Tracking-Q initialization requires the expert trajectory buffer")
        motion_ids = [int(motion_id) for motion_id in expert_buffer.motion_ids]
        if motion_ids != list(range(len(motion_ids))):
            raise ValueError("Tracking-Q expert trajectories must use global motion ids 0..N-1")
        lengths = torch.tensor(
            [expert_buffer.motion_length(motion_id) for motion_id in motion_ids],
            dtype=torch.long,
        )
        if self.cfg.pace is not None:
            if self._pace_roles is None:
                raise RuntimeError("PACE roles must be initialized before tracking Q")
            roles = self._pace_roles.copy()
        else:
            num_envs = int(self.cfg.online_parallel_envs)
            rng = np.random.default_rng(int(self.cfg.seed))
            permutation = rng.permutation(num_envs)
            roles = np.zeros(num_envs, dtype=bool)
            roles[permutation[: num_envs // 2]] = True
        self._tracking_roles = roles
        train_env._env.configure_tracking_rollouts(
            roles,
            expert_motion_lengths=lengths,
            seq_length=int(self.cfg.agent.model.seq_length),
            expert_rollout_max_steps=int(self.cfg.agent.train.rollout_expert_trajectories_length),
        )
        if self._write_shared_artifacts:
            print(
                "[Tracking Q] Fixed environment roles: "
                f"aligned_expert={int(roles.sum())}, random_or_mixed={int((~roles).sum())}, "
                "aligned_rollout=full_motion_from_frame_zero",
                flush=True,
            )

    def _apply_pace_actor_schedule(self) -> None:
        if self.cfg.pace is None:
            return
        completed = int(self._pace_windows_completed)
        warmup = int(self.cfg.pace.actor_warmup_windows)
        enabled = completed >= warmup
        if enabled:
            ramp_index = completed - warmup + 1
            lr_multiplier = min(ramp_index / float(self.cfg.pace.actor_lr_ramp_windows), 1.0)
        else:
            lr_multiplier = 0.0
        if self._pace_retention_recovery:
            lr_multiplier *= 0.25
            acquisition_loss_weight = 0.0
        elif self._pace_retention_warning:
            lr_multiplier *= 0.5
            acquisition_loss_weight = 1.0
        else:
            acquisition_loss_weight = 1.0
        self.agent.set_pace_actor_schedule(
            enabled=enabled,
            lr_multiplier=lr_multiplier,
            acquisition_loss_weight=acquisition_loss_weight,
        )

    def _pace_transition_fields(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pace_roles is None or self._pace_alive is None:
            raise RuntimeError("PACE runtime roles have not been initialized")
        return self._pace_roles.copy(), self._pace_alive.copy()

    def _update_agent_pace_metrics(
        self,
        *,
        valid_acquisition: int,
        valid_consolidation: int,
        total_acquisition: int,
        total_consolidation: int,
    ) -> None:
        metrics = {
            "pace/lambda_con": float(self._pace_lambda_con),
            "pace/rho": float(self._pace_latest_rho),
            "pace/rho_ema": float(self._pace_rho_ema),
            "pace/valid_acquisition": float(valid_acquisition),
            "pace/valid_consolidation": float(valid_consolidation),
            "pace/valid_ratio_acquisition": valid_acquisition / max(total_acquisition, 1),
            "pace/valid_ratio_consolidation": valid_consolidation / max(total_consolidation, 1),
            "pace/windows_completed": float(self._pace_windows_completed),
            "pace/actor_warmup_active": float(not self.agent._pace_actor_updates_enabled),
            "pace/actor_lr_multiplier": float(self.agent._pace_actor_lr_multiplier),
            "pace/retention_recovery": float(self._pace_retention_recovery),
            "pace/retention_warning": float(self._pace_retention_warning),
        }
        self.agent.set_pace_runtime_metrics(metrics)

    def _complete_pace_window(self) -> None:
        if self.cfg.pace is None:
            return
        counts = torch.tensor(
            [
                self._pace_valid_acquisition,
                self._pace_valid_consolidation,
                self._pace_total_acquisition,
                self._pace_total_consolidation,
            ],
            dtype=torch.float64,
            device=self.agent.device,
        )
        if self.cfg.distributed_sync:
            import torch.distributed as dist

            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        valid_acquisition, valid_consolidation, total_acquisition, total_consolidation = [
            int(value) for value in counts.detach().cpu().tolist()
        ]
        denominator = valid_acquisition + valid_consolidation
        if denominator <= 0:
            raise RuntimeError("PACE completed a rollout window without any valid samples")
        rho = valid_acquisition / denominator
        self._pace_latest_rho = rho
        self._pace_rho_ema = (
            self.cfg.pace.rho_ema_beta * self._pace_rho_ema
            + (1.0 - self.cfg.pace.rho_ema_beta) * rho
        )
        self._pace_lambda_con = min(
            self.cfg.pace.lambda_max,
            self.cfg.pace.lambda_base
            + self.cfg.pace.lambda_gain * max(0.0, self._pace_rho_ema - self.cfg.pace.rho_reference),
        )
        self._pace_windows_completed += 1
        self.agent.set_pace_lambda(self._pace_lambda_con)
        self._apply_pace_actor_schedule()
        self._update_agent_pace_metrics(
            valid_acquisition=valid_acquisition,
            valid_consolidation=valid_consolidation,
            total_acquisition=total_acquisition,
            total_consolidation=total_consolidation,
        )
        if self._write_shared_artifacts:
            print(
                "[PACE] "
                f"window={self._pace_windows_completed} NA={valid_acquisition} NC={valid_consolidation} "
                f"rho={rho:.6f} rho_ema={self._pace_rho_ema:.6f} lambda_con={self._pace_lambda_con:.6f}",
                flush=True,
            )
        self._pace_window_step = 0
        self._pace_valid_acquisition = 0
        self._pace_valid_consolidation = 0
        self._pace_total_acquisition = 0
        self._pace_total_consolidation = 0
        if self._pace_alive is not None:
            # Acquisition failures remain latched until that full expert
            # motion ends; consolidation has no motion-tracking failure latch.
            self._pace_alive[~self._pace_roles] = True

    def _record_pace_step(self, new_info: dict[str, tp.Any]) -> None:
        if self.cfg.pace is None or self._pace_roles is None or self._pace_alive is None:
            return
        self._pace_valid_acquisition += int(np.logical_and(self._pace_alive, self._pace_roles).sum())
        self._pace_valid_consolidation += int(np.logical_and(self._pace_alive, ~self._pace_roles).sum())
        self._pace_total_acquisition += int(self._pace_roles.sum())
        self._pace_total_consolidation += int((~self._pace_roles).sum())
        pace_failure = np.asarray(new_info.get("pace_failure", np.zeros_like(self._pace_roles)), dtype=bool).reshape(-1)
        if pace_failure.shape != self._pace_roles.shape:
            raise ValueError(f"PACE failure mask has shape {pace_failure.shape}, expected {self._pace_roles.shape}")
        self._pace_alive[np.logical_and(self._pace_roles, pace_failure)] = False
        pace_segment_end = np.asarray(
            new_info.get("pace_segment_end", np.zeros_like(self._pace_roles)), dtype=bool
        ).reshape(-1)
        if pace_segment_end.shape != self._pace_roles.shape:
            raise ValueError(
                f"PACE segment-end mask has shape {pace_segment_end.shape}, expected {self._pace_roles.shape}"
            )
        # The environment has already reset these slots to expert frame zero;
        # their next transition starts a fresh valid full-motion rollout.
        self._pace_alive[np.logical_and(self._pace_roles, pace_segment_end)] = True
        self._pace_window_step += 1
        if self._pace_window_step >= self.cfg.pace.rollout_horizon:
            self._complete_pace_window()

    def _discard_partial_pace_window(self) -> None:
        """Start a fresh validity window after an external full-env reset."""
        if self.cfg.pace is None:
            return
        self._pace_window_step = 0
        self._pace_valid_acquisition = 0
        self._pace_valid_consolidation = 0
        self._pace_total_acquisition = 0
        self._pace_total_consolidation = 0
        if self._pace_alive is not None:
            self._pace_alive[:] = True

    def _update_pace_retention_state(self, grouped: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Update retention guards from EMDs on the immutable base-policy split."""
        if self.cfg.pace is None:
            return {}
        group_ids = grouped["group_ids"].detach().cpu().long()
        if self._pace_fixed_group_ids is None:
            self._pace_fixed_group_ids = group_ids.clone()
        elif not torch.equal(self._pace_fixed_group_ids, group_ids):
            raise RuntimeError("PACE mastered/challenging membership changed after the base-policy split")

        stats = grouped["stats"]
        mastered_mean = float(stats["eval/pace_sampling/mastered/mean_emd"])
        challenging_mean = float(stats["eval/pace_sampling/challenging/mean_emd"])
        mastered_count = int(stats["eval/pace_sampling/mastered/count"])
        challenging_count = int(stats["eval/pace_sampling/challenging/count"])
        global_mean = (
            mastered_mean * mastered_count + challenging_mean * challenging_count
        ) / max(mastered_count + challenging_count, 1)

        if self._pace_baseline_emd is None:
            self._pace_baseline_emd = {
                "global": global_mean,
                "mastered": mastered_mean,
                "challenging": challenging_mean,
            }
            self._pace_best_mastered_emd = mastered_mean
        baseline_mastered = max(float(self._pace_baseline_emd["mastered"]), 1.0e-12)
        mastered_ratio = mastered_mean / baseline_mastered
        self._pace_best_mastered_emd = min(
            mastered_mean,
            float(self._pace_best_mastered_emd if self._pace_best_mastered_emd is not None else mastered_mean),
        )
        self._pace_retention_warning = mastered_ratio > self.cfg.pace.retention_warn_ratio
        if mastered_ratio > self.cfg.pace.retention_recovery_ratio:
            self._pace_retention_violation_count += 1
        else:
            self._pace_retention_violation_count = 0
            if mastered_ratio <= self.cfg.pace.retention_warn_ratio:
                self._pace_retention_recovery = False
        if self._pace_retention_violation_count >= self.cfg.pace.retention_recovery_evals:
            self._pace_retention_recovery = True
        self._pace_retention_stop_requested = mastered_ratio > self.cfg.pace.retention_stop_ratio

        stats.update(
            {
                "eval/pace_retention/baseline_global_emd": float(self._pace_baseline_emd["global"]),
                "eval/pace_retention/baseline_mastered_emd": float(self._pace_baseline_emd["mastered"]),
                "eval/pace_retention/current_global_emd": global_mean,
                "eval/pace_retention/current_mastered_emd": mastered_mean,
                "eval/pace_retention/mastered_emd_ratio": mastered_ratio,
                "eval/pace_retention/best_mastered_emd": float(self._pace_best_mastered_emd),
                "eval/pace_retention/warning": float(self._pace_retention_warning),
                "eval/pace_retention/recovery": float(self._pace_retention_recovery),
                "eval/pace_retention/stop_requested": float(self._pace_retention_stop_requested),
            }
        )
        if self._write_shared_artifacts and self._pace_retention_warning:
            print(
                "[PACE][RETENTION] "
                f"mastered EMD={mastered_mean:.6f} is {mastered_ratio:.3f}x the base-policy value "
                f"{baseline_mastered:.6f}; recovery={self._pace_retention_recovery}, "
                f"stop={self._pace_retention_stop_requested}",
                flush=True,
            )
        return {
            "fixed_group_ids": self._pace_fixed_group_ids,
            "baseline_emd": self._pace_baseline_emd,
            "best_mastered_emd": self._pace_best_mastered_emd,
            "retention_violation_count": self._pace_retention_violation_count,
            "retention_recovery": self._pace_retention_recovery,
            "retention_warning": self._pace_retention_warning,
            "retention_stop_requested": self._pace_retention_stop_requested,
        }

    def _apply_pace_retention_payload(self, state: dict[str, tp.Any]) -> None:
        if not state:
            return
        self._pace_fixed_group_ids = torch.as_tensor(state["fixed_group_ids"], dtype=torch.long).clone()
        self._pace_baseline_emd = {
            str(key): float(value) for key, value in state["baseline_emd"].items()
        }
        self._pace_best_mastered_emd = float(state["best_mastered_emd"])
        self._pace_retention_violation_count = int(state["retention_violation_count"])
        self._pace_retention_recovery = bool(state["retention_recovery"])
        self._pace_retention_warning = bool(state["retention_warning"])
        self._pace_retention_stop_requested = bool(state["retention_stop_requested"])
        self._apply_pace_actor_schedule()
        self._update_agent_pace_metrics(
            valid_acquisition=self._pace_valid_acquisition,
            valid_consolidation=self._pace_valid_consolidation,
            total_acquisition=self._pace_total_acquisition,
            total_consolidation=self._pace_total_consolidation,
        )

    def _save_pace_eval_snapshot(self, global_time: int) -> None:
        """Save a compact actor-only, versioned snapshot at every full PACE eval."""
        if self.cfg.pace is None:
            return
        if self.cfg.distributed_sync:
            barrier()
        if self._write_shared_artifacts:
            snapshot_dir = self.work_dir / "pace_eval_checkpoints" / f"step_{int(global_time):012d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(
                self.agent._model._actor.state_dict(),
                str(snapshot_dir / "actor.safetensors"),
            )
            with (snapshot_dir / "pace_status.json").open("w+") as file:
                json.dump(self._pace_checkpoint_state(), file, indent=4)
        if self.cfg.distributed_sync:
            barrier()

    def _pace_checkpoint_state(self) -> dict[str, tp.Any] | None:
        if self.cfg.pace is None:
            return None
        return {
            "latest_rho": float(self._pace_latest_rho),
            "rho_ema": float(self._pace_rho_ema),
            "lambda_con": float(self._pace_lambda_con),
            "windows_completed": int(self._pace_windows_completed),
            "fixed_group_ids": (
                self._pace_fixed_group_ids.tolist() if self._pace_fixed_group_ids is not None else None
            ),
            "baseline_emd": self._pace_baseline_emd,
            "best_mastered_emd": self._pace_best_mastered_emd,
            "retention_violation_count": int(self._pace_retention_violation_count),
            "retention_recovery": bool(self._pace_retention_recovery),
            "retention_warning": bool(self._pace_retention_warning),
            "actor_runtime": self.agent.pace_checkpoint_state(),
        }

    def train_online(self) -> None:
        if self.training_with_expert_data:
            if self.cfg.load_expert_data_from_motion_lib:
                expert_buffer = load_expert_trajectories_from_motion_lib(
                    self.train_env._env,
                    self.cfg.agent,
                    device=self.cfg.expert_buffer_device,
                )
            else:
                raise RuntimeError(
                    "This MJLab-focused build only supports expert data loaded from the motion library. "
                    "Set load_expert_data_from_motion_lib=True."
                )
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        checkpoint_buffer_dir = self._checkpoint_buffer_path(checkpoint_dir)
        if checkpoint_buffer_dir.exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=_trajectory_output_keys(self.cfg.agent)
                    + (["pace_role", "pace_valid"] if self.cfg.pace is not None else []),
                    output_key_tp1=["observation", "terminated"]
                    + (
                        ["z", "reference_motion_id", "reference_step"]
                        if bool(getattr(self.cfg.agent.model, "use_tracking_critic", False))
                        else []
                    ),
                )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer
        self._initialize_pace_runtime(train_env, expert_buffer if self.training_with_expert_data else None)
        self._initialize_tracking_runtime(train_env, expert_buffer if self.training_with_expert_data else None)

        print("Starting training")
        global_step_scale = _global_step_scale(self.cfg)
        local_step_increment = self.cfg.online_parallel_envs
        global_step_increment = local_step_increment * global_step_scale
        max_local_time = (self.cfg.num_env_steps + global_step_scale - 1) // global_step_scale
        if self._write_shared_artifacts:
            print(
                "[INFO] Step accounting: "
                f"num_envs_per_rank={_num_envs_per_rank(self.cfg)}, global_parallel_envs={_global_parallel_envs(self.cfg)}, "
                f"local_step_increment={local_step_increment}, global_step_increment={global_step_increment}, "
                f"num_env_steps_global={self.cfg.num_env_steps}, num_seed_steps_local={self.cfg.num_seed_steps}, "
                f"update_agent_every_local={self.cfg.update_agent_every}, world_size={self.distributed_world_size}, "
                f"loss_mode={_distributed_loss_mode(self.cfg)}, effective_batch_size={_effective_batch_size(self.cfg)}, "
                f"replay_capacity_per_rank={_replay_capacity_per_rank(self.cfg)}, "
                f"effective_replay_capacity={_effective_replay_capacity(self.cfg)}, "
                f"trajectory_steps_per_rank={_trajectory_steps_per_rank(self.cfg)}, "
                f"compile={self.cfg.agent.compile}"
            )
        progb = tqdm(
            total=self.cfg.num_env_steps,
            initial=min(self._checkpoint_global_time, self.cfg.num_env_steps),
            disable=self.cfg.disable_tqdm,
        )
        td, info = train_env.reset()
        if self.cfg.fail_on_nonfinite:
            _assert_finite(
                td,
                label="env.reset.obs",
                rank=self.distributed_rank,
                local_time=self._checkpoint_local_time,
                global_time=self._checkpoint_global_time,
                optimizer_steps=self._optimizer_steps,
            )
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        total_metrics, context = None, None
        metric_update_counts: dict[str, int] = {}
        start_time = time.time()
        collection_time_acc = 0.0
        learning_time_acc = 0.0
        steps_since_last_log = 0
        train_time_total = 0.0
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_local_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.log_every_updates)

        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        for local_time in range(self._checkpoint_local_time, max_local_time + local_step_increment, local_step_increment):
            global_time = local_time * global_step_scale
            if global_time > self.cfg.num_env_steps:
                break
            if (local_time != self._checkpoint_local_time) and checkpoint_time_checker.check(global_time):
                checkpoint_time_checker.update_last_step(global_time)
                self.save(local_time=local_time, global_time=global_time, optimizer_steps=self._optimizer_steps, replay_buffer=replay_buffer)

            if global_time >= self.cfg.num_env_steps:
                break

            if (self.evaluate and eval_time_checker.check(global_time)) or (
                self.evaluate and global_time == self._checkpoint_global_time
            ):
                eval_metrics = {}
                run_eval_on_this_rank = (not self.cfg.distributed_sync) or self.distributed_rank == 0
                if run_eval_on_this_rank:
                    eval_metrics = self.eval(global_time, replay_buffer=replay_buffer)
                if self.cfg.distributed_sync:
                    barrier()
                eval_time_checker.update_last_step(global_time)
                grouped_single_source = False
                if self.cfg.prioritization:
                    # priorities
                    priority_payload = None
                    if run_eval_on_this_rank:
                        grouped_single_source = train_env._env._motion_lib.num_motion_sources == 1
                        if not grouped_single_source:
                            assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                                "Mismatch in number of motions returned by the eval"
                            )
                        index_in_buffer, name_in_buffer = {}, {}
                        for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                            index_in_buffer[motion_id] = i
                            if hasattr(replay_buffer["expert_slicer"], "file_names"):
                                name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                        if grouped_single_source:
                            if self.cfg.prioritization_mode != "exp":
                                raise ValueError(
                                    "Single-source global EMD sampling requires prioritization_mode='exp' to preserve the configured formula"
                                )
                            global_emd = _compute_global_emd_distribution(
                                eval_metrics[self.priorization_eval_name],
                                list(replay_buffer["expert_slicer"].motion_ids),
                                clamp_min=self.cfg.prioritization_min_val,
                                clamp_max=self.cfg.prioritization_max_val,
                                scale=self.cfg.prioritization_scale,
                            )
                            priority_payload = {"mode": "global_emd", **global_emd}
                        else:
                            motions_id, priorities, idxs = [], [], []
                            for _, metr in eval_metrics[self.priorization_eval_name].items():
                                motions_id.append(metr["motion_id"])
                                priorities.append(metr["emd"])
                                idxs.append(index_in_buffer[metr["motion_id"]])
                            priorities = (
                                torch.clamp(
                                    torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                                    min=self.cfg.prioritization_min_val,
                                    max=self.cfg.prioritization_max_val,
                                )
                                * self.cfg.prioritization_scale
                            )

                            if self.cfg.prioritization_mode == "lin":
                                pass
                            elif self.cfg.prioritization_mode == "exp":
                                priorities = 2**priorities
                            elif self.cfg.prioritization_mode == "bin":
                                bins = torch.floor(priorities)
                                for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                                    mask = bins == i
                                    n = mask.sum().item()
                                    if n > 0:
                                        priorities[mask] = 1 / n
                            else:
                                raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
                            priority_payload = {
                                "mode": "legacy_sources",
                                "priorities": priorities.detach().cpu(),
                                "idxs": idxs,
                                "file_name": name_in_buffer,
                            }
                    if self.cfg.distributed_sync:
                        priority_payload = broadcast_object(priority_payload, src=0)
                    if priority_payload is None:
                        raise RuntimeError("Prioritization requires evaluation metrics, but no priority payload was produced.")
                    grouped_single_source = priority_payload["mode"] == "global_emd"
                    if grouped_single_source:
                        probabilities = priority_payload["probabilities"]
                        train_env._env._motion_lib.set_global_emd_sampling(probabilities)
                        expert_device = replay_buffer["expert_slicer"].priorities.device
                        replay_buffer["expert_slicer"].set_global_sampling(probabilities.to(expert_device))
                        if self._write_shared_artifacts:
                            _log_emd_group_stats(
                                priority_payload["stats"],
                                timestep=global_time,
                                use_wandb=bool(self.cfg.use_wandb),
                                writer=self.tb_writer,
                            )
                    else:
                        priorities = priority_payload["priorities"].to(self.agent.device)
                        idxs = priority_payload["idxs"]
                        name_in_buffer = priority_payload["file_name"]
                        train_env._env._motion_lib.update_sampling_weight_by_id(
                            priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                        )
                        expert_device = replay_buffer["expert_slicer"].priorities.device
                        replay_buffer["expert_slicer"].update_priorities(
                            priorities=priorities.to(expert_device),
                            idxs=torch.tensor(np.array(idxs), device=expert_device),
                        )

                if self.cfg.pace is not None:
                    self._save_pace_eval_snapshot(global_time)
                    if self._pace_retention_stop_requested:
                        raise RuntimeError(
                            "PACE retention guard stopped training: mastered-set EMD exceeded "
                            f"{self.cfg.pace.retention_stop_ratio:.2f}x its immutable base-policy baseline. "
                            "An actor-only snapshot for this eval was saved under pace_eval_checkpoints/."
                        )

                if uses_humanoidverse_eval:
                    # Evaluation leaves the last chunk loaded. Install any newly
                    # computed priorities first, then reload one training pool on
                    # every rank. The wrapper reset below is the sole post-eval
                    # reset, avoiding an old-priority reload and duplicate resets.
                    train_env._env.prepare_training_motion_pool()
                    td, info = train_env.reset()
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            td,
                            label="env.post_eval_reset.obs",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    self._discard_partial_pace_window()
                    context = None

            if global_time + global_step_increment > self.cfg.num_env_steps:
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Stopping before next rollout to avoid exceeding global sample budget: "
                        f"current_global_time={global_time}, next_global_time={global_time + global_step_increment}, "
                        f"num_env_steps_global={self.cfg.num_env_steps}"
                    )
                break

            rollout_start = time.time()
            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                if self.cfg.pace is not None:
                    rollout_metadata = train_env._env.get_pace_rollout_metadata()
                elif bool(getattr(self.cfg.agent.model, "use_tracking_critic", False)):
                    rollout_metadata = train_env._env.get_tracking_rollout_metadata()
                else:
                    rollout_metadata = None
                context = self.agent.maybe_update_rollout_context(
                    z=context,
                    step_count=step_count,
                    replay_buffer=replay_buffer,
                    rollout_metadata=rollout_metadata,
                )
                if (
                    local_time < self.cfg.num_seed_steps
                    and self.cfg.pace is None
                    and not bool(getattr(self.cfg.agent.model, "use_tracking_critic", False))
                ):
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                        action = action.cpu().detach().numpy()
                check_rollout_nonfinite = (
                    self.cfg.fail_on_nonfinite
                    and self.cfg.nonfinite_check_rollout_every_local_steps > 0
                    and local_time % self.cfg.nonfinite_check_rollout_every_local_steps == 0
                )
                if check_rollout_nonfinite:
                    _assert_finite(
                        obs,
                        label="rollout.obs",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        context,
                        label="rollout.context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        history_context,
                        label="rollout.history_context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        action,
                        label="rollout.action",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
            pace_role, pace_valid = self._pace_transition_fields() if self.cfg.pace is not None else (None, None)
            tracking_role = None
            reference_motion_id = None
            reference_step = None
            if bool(getattr(self.cfg.agent.model, "use_tracking_critic", False)):
                if rollout_metadata is None:
                    raise RuntimeError("Tracking-Q rollout metadata is missing")
                role_key = "acquisition" if self.cfg.pace is not None else "expert_rollout"
                tracking_role_t = rollout_metadata[role_key].to(self.agent.device).bool()
                reference_motion_id_t = rollout_metadata["global_motion_ids"].to(self.agent.device).long()
                reference_step_t = (
                    rollout_metadata["motion_start_steps"].to(self.agent.device).long()
                    + step_count.reshape(-1).to(self.agent.device).long()
                    + 1
                )
                sentinel = torch.full_like(reference_motion_id_t, -1)
                tracking_role = tracking_role_t.detach().cpu().numpy()
                reference_motion_id = torch.where(
                    tracking_role_t, reference_motion_id_t, sentinel
                ).detach().cpu().numpy().astype(np.int32)
                reference_step = torch.where(
                    tracking_role_t, reference_step_t, sentinel
                ).detach().cpu().numpy().astype(np.int32)
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
            self._record_pace_step(new_info)
            self._record_fixed_goal_progress(
                next_observation=new_td,
                target_z=context,
                step_count=step_count,
                terminated=new_terminated,
                truncated=new_truncated,
            )
            collection_time_acc += time.time() - rollout_start
            steps_since_last_log += global_step_increment
            if check_rollout_nonfinite:
                _assert_finite(
                    new_td,
                    label="env.step.obs",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                _assert_finite(
                    new_reward,
                    label="env.step.reward",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )

            # we check if at the next iteration we will evaluate
            next_local_time = local_time + local_step_increment
            next_global_time = next_local_time * global_step_scale
            if (self.evaluate and eval_time_checker.check(next_global_time)) or (
                self.evaluate and next_global_time == self._checkpoint_global_time
            ):
                if isinstance(self.cfg.env, HumanoidVerseMjlabConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if pace_role is not None and pace_valid is not None:
                        data["pace_role"] = pace_role[None, ..., None]
                        data["pace_valid"] = pace_valid[None, ..., None]
                    if tracking_role is not None:
                        data["expert_z_rollout"] = tracking_role[None, ..., None]
                        data["reference_motion_id"] = reference_motion_id[None, ..., None]
                        data["reference_step"] = reference_step[None, ..., None]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                    if "tracking_rewards" in new_info:
                        data["tracking_rewards"] = {
                            k: v[None, ..., None]
                            for k, v in new_info["tracking_rewards"].items()
                            if not k.startswith("_")
                        }
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done

                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            if check_rollout_nonfinite:
                _assert_finite(
                    data,
                    label="replay.extend.data",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
            replay_buffer["train"].extend(data)

            warmup_complete = local_time + local_step_increment >= self.cfg.num_seed_steps
            if len(replay_buffer["train"]) > 0 and warmup_complete and update_agent_time_checker.check(local_time):
                update_agent_time_checker.update_last_step(local_time)
                learning_start = time.time()
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, local_time)
                    self._optimizer_steps += 1
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            metrics,
                            label="agent.update.metrics",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                        if (
                            self.cfg.nonfinite_check_model_every_updates > 0
                            and self._optimizer_steps % self.cfg.nonfinite_check_model_every_updates == 0
                        ):
                            _assert_model_finite(
                                self.agent._model,
                                rank=self.distributed_rank,
                                local_time=local_time,
                                global_time=global_time,
                                optimizer_steps=self._optimizer_steps,
                            )
                    if self.cfg.distributed_sync:
                        sync_floating_buffers(self.agent._model)
                    if self.cfg.distributed_sync and self.cfg.distributed_average_metrics:
                        metrics = average_metrics(metrics)
                    total_metrics, metric_update_counts = _accumulate_metrics(
                        total_metrics,
                        metric_update_counts,
                        metrics,
                    )
                learning_time_acc += time.time() - learning_start

            if log_time_checker.check(global_time) and total_metrics is not None:
                log_time_checker.update_last_step(global_time)
                current_global_time = min(global_time + global_step_increment, self.cfg.num_env_steps)
                current_local_time = local_time + local_step_increment
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / metric_update_counts[k]
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict.update(self._get_torso_contact_force_metrics(train_env))
                m_dict.update(self._consume_fixed_goal_probe_metrics())
                if self.cfg.distributed_sync and self.distributed_world_size > 1:
                    m_dict["distributed/world_size"] = self.distributed_world_size
                    m_dict["distributed/effective_batch_size"] = _effective_batch_size(self.cfg)
                    m_dict["distributed/num_envs_per_rank"] = _num_envs_per_rank(self.cfg)
                    m_dict["distributed/global_parallel_envs"] = _global_parallel_envs(self.cfg)
                    m_dict["distributed/replay_capacity_per_rank"] = _replay_capacity_per_rank(self.cfg)
                    m_dict["distributed/effective_replay_capacity"] = _effective_replay_capacity(self.cfg)
                    m_dict["distributed/trajectory_steps_per_rank"] = _trajectory_steps_per_rank(self.cfg)
                    m_dict["distributed/compile"] = int(bool(self.cfg.agent.compile))
                m_dict["distributed/local_env_steps"] = int(current_local_time)
                m_dict["distributed/global_env_steps"] = int(current_global_time)
                m_dict["distributed/optimizer_steps"] = int(self._optimizer_steps)

                interval_train_time = collection_time_acc + learning_time_acc
                train_time_total += interval_train_time
                interval_fps = steps_since_last_log / max(interval_train_time, 1e-9)
                total_train_steps = max(current_global_time - self._checkpoint_global_time, 0)
                avg_train_fps = total_train_steps / max(train_time_total, 1e-9) if total_train_steps > 0 else interval_fps
                current_iteration = current_global_time // global_step_increment
                max_iterations = (self.cfg.num_env_steps + global_step_increment - 1) // global_step_increment
                elapsed_time = time.time() - start_time
                remaining_steps = max(self.cfg.num_env_steps - current_global_time, 0)
                eta_seconds = remaining_steps / max(avg_train_fps, 1e-9) if remaining_steps > 0 else 0.0
                grouped_metrics = _build_grouped_metrics(m_dict)

                if self._write_shared_artifacts and self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{key}": value for _, key, value in _iter_tracked_group_metrics(grouped_metrics)},
                        step=current_global_time,
                    )
                if self._write_shared_artifacts:
                    log_text = _render_grouped_train_log(
                        grouped_metrics=grouped_metrics,
                        timestep=current_global_time,
                        iteration=current_iteration,
                        max_iterations=max_iterations,
                        steps_per_second=interval_fps,
                        collection_time=collection_time_acc,
                        learning_time=learning_time_acc,
                        elapsed_time=elapsed_time,
                        eta_seconds=eta_seconds,
                    )
                    print(log_text, flush=True)
                _log_grouped_tensorboard(
                    self.tb_writer,
                    grouped_metrics,
                    current_global_time,
                    steps_per_second=interval_fps,
                    collection_time=collection_time_acc,
                    learning_time=learning_time_acc,
                    elapsed_time=elapsed_time,
                    eta_seconds=eta_seconds,
                    iteration=current_iteration,
                )
                total_metrics = None
                metric_update_counts = {}
                collection_time_acc = 0.0
                learning_time_acc = 0.0
                steps_since_last_log = 0
                m_dict["timestep"] = current_global_time
                m_dict["local_timestep"] = current_local_time
                m_dict["iteration"] = current_iteration
                if self.train_logger is not None:
                    self.train_logger.log(m_dict)

            progb.update(global_step_increment)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}
        if hasattr(self.agent, "set_train_mode"):
            self.agent.set_train_mode(False)
        else:
            self.agent._model.train(False)
        self._log_b_convergence(timestep=t, replay_buffer=replay_buffer)

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers.get(evaluation_name)
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                self.agent._model.to("cpu")
            if hasattr(self.agent, "set_train_mode"):
                self.agent.set_train_mode(False)
            else:
                self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {
                        f"eval/{evaluation_name}/{k}": scalar
                        for k, v in wandb_dict.items()
                        if (scalar := _as_python_scalar(v)) is not None
                    },
                    step=t,
                )
            if wandb_dict is not None and self.tb_writer is not None:
                for key, value in wandb_dict.items():
                    scalar = _as_python_scalar(value)
                    if scalar is not None:
                        self.tb_writer.add_scalar(f"eval/{evaluation_name}/{key}", scalar, t)
                self.tb_writer.flush()

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        if hasattr(self.agent, "set_train_mode"):
            self.agent.set_train_mode(True)
        else:
            self.agent._model.train()

        return evaluation_results

    def save(self, *, local_time: int, global_time: int, optimizer_steps: int, replay_buffer: Dict[str, tp.Any]) -> None:
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        sync_report = None
        if self.cfg.distributed_sync:
            barrier()
            sync_report = module_sync_report(self.agent._model, src=0)
            if sync_report["max_abs_diff_from_rank0"] > 1.0e-5:
                raise RuntimeError(f"Distributed model state diverged before checkpoint: {sync_report}")
        if self._write_shared_artifacts:
            print(f"Checkpointing at local_time={local_time} global_time={global_time} optimizer_steps={optimizer_steps}")
            self.agent.save(str(checkpoint_dir))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self._checkpoint_buffer_path(checkpoint_dir))
        if self.cfg.distributed_sync:
            barrier()
        if self._write_shared_artifacts:
            if sync_report is not None:
                with (checkpoint_dir / "distributed_sync.json").open("w+") as f:
                    json.dump(sync_report, f, indent=4)
            with (checkpoint_dir / "train_status.json").open("w+") as f:
                json.dump(
                    _make_train_status(
                        self.cfg,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=optimizer_steps,
                        pace_state=self._pace_checkpoint_state(),
                        tracking_q_state=(
                            self.agent.tracking_checkpoint_state()
                            if bool(getattr(self.cfg.agent.model, "use_tracking_critic", False))
                            else None
                        ),
                    ),
                    f,
                    indent=4,
                )
        if self.cfg.distributed_sync:
            barrier()


def train_bfm_zero():
    raise RuntimeError(
        "Legacy train_bfm_zero entrypoint is disabled in this MJLab build. "
        "Use humanoidverse.train or ./run_train.sh."
    )

if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_bfm_zero()

# uv run --no-cache -m humanoidverse.meta_online_entry_point
