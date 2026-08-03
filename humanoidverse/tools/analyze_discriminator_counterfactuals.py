"""Training-free counterfactual tests for discriminator state/z reliance."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from humanoidverse.agents.envs.expert_motion_loader import (
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training-free state/z counterfactual discriminator tests."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-rank", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _recompute_z(
    model: torch.nn.Module,
    future_state: np.ndarray,
    future_privileged: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    num_samples, seq_length = future_state.shape[:2]
    obs = {
        "state": torch.from_numpy(future_state.reshape(-1, future_state.shape[-1])).to(device),
        "privileged_state": torch.from_numpy(
            future_privileged.reshape(-1, future_privileged.shape[-1])
        ).to(device),
    }
    backward = model.backward_map(obs)
    z = backward.view(num_samples, seq_length, backward.shape[-1]).mean(dim=1)
    return model.project_z(z).float().cpu().numpy()


def _sample_policy_windows(
    model: torch.nn.Module,
    buffer_paths: list[Path],
    *,
    samples_per_rank: int,
    seq_length: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    states: list[np.ndarray] = []
    privileged_states: list[np.ndarray] = []
    stored_zs: list[np.ndarray] = []
    recomputed_zs: list[np.ndarray] = []
    rank_metadata: list[dict[str, Any]] = []

    for rank, buffer_path in enumerate(buffer_paths):
        config = _read_json(buffer_path.parent / "config.json")
        cursor = int(config["_idx"])
        with h5py.File(buffer_path, "r") as file:
            capacity, num_envs = file["observation-state"].shape[:2]
            candidates = np.arange(0, capacity - seq_length, dtype=np.int64)
            crosses_cursor = (candidates < cursor) & (
                candidates + seq_length >= cursor
            )
            candidates = candidates[~crosses_cursor]
            rng.shuffle(candidates)

            rank_state: list[np.ndarray] = []
            rank_privileged: list[np.ndarray] = []
            rank_stored_z: list[np.ndarray] = []
            rank_recomputed_z: list[np.ndarray] = []
            used_starts: list[int] = []
            collected = 0
            for start in candidates:
                if collected >= samples_per_rank:
                    break
                start = int(start)
                truncated = file["truncated"][start : start + seq_length, :, 0]
                valid = ~np.any(truncated, axis=0)
                if not np.any(valid):
                    continue

                current_state = file["observation-state"][start][valid]
                current_privileged = file["observation-privileged_state"][start][valid]
                current_z = file["z"][start][valid]
                future_state = file["observation-state"][
                    start + 1 : start + seq_length + 1
                ][:, valid].transpose(1, 0, 2)
                future_privileged = file["observation-privileged_state"][
                    start + 1 : start + seq_length + 1
                ][:, valid].transpose(1, 0, 2)
                recomputed_z = _recompute_z(
                    model,
                    future_state,
                    future_privileged,
                    device=device,
                )

                take = min(samples_per_rank - collected, current_state.shape[0])
                rank_state.append(current_state[:take])
                rank_privileged.append(current_privileged[:take])
                rank_stored_z.append(current_z[:take])
                rank_recomputed_z.append(recomputed_z[:take])
                collected += take
                used_starts.append(start)

        if collected != samples_per_rank:
            raise RuntimeError(
                f"Could only collect {collected}/{samples_per_rank} valid samples from rank {rank}"
            )
        state_array = np.concatenate(rank_state)
        privileged_array = np.concatenate(rank_privileged)
        stored_array = np.concatenate(rank_stored_z)
        recomputed_array = np.concatenate(rank_recomputed_z)
        states.append(state_array)
        privileged_states.append(privileged_array)
        stored_zs.append(stored_array)
        recomputed_zs.append(recomputed_array)
        rank_metadata.append(
            {
                "rank": rank,
                "buffer": str(buffer_path),
                "cursor": cursor,
                "used_window_starts": used_starts,
                "samples": collected,
            }
        )
        print(
            f"[CF-DISC] policy rank={rank} samples={collected} windows={len(used_starts)}",
            flush=True,
        )

    return (
        {
            "state": np.concatenate(states),
            "privileged_state": np.concatenate(privileged_states),
        },
        np.concatenate(stored_zs),
        np.concatenate(recomputed_zs),
        rank_metadata,
    )


@torch.no_grad()
def _sample_expert(
    model: torch.nn.Module,
    expert_buffer: Any,
    *,
    num_samples: int,
    seq_length: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    batch = expert_buffer.sample(num_samples, seq_length=seq_length)
    obs = batch["observation"]
    next_obs = batch["next"]["observation"]
    backward = model.backward_map(next_obs)
    z = backward.view(num_samples // seq_length, seq_length, -1).mean(dim=1)
    z = model.project_z(z)
    z = torch.repeat_interleave(z, seq_length, dim=0)
    return (
        {
            "state": obs["state"].float().cpu().numpy(),
            "privileged_state": obs["privileged_state"].float().cpu().numpy(),
        },
        z.float().cpu().numpy(),
    )


@torch.no_grad()
def _evaluate_pair(
    model: torch.nn.Module,
    obs: dict[str, np.ndarray],
    z: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        obs_batch = {
            key: torch.from_numpy(value[start:end]).to(device)
            for key, value in obs.items()
        }
        normalized_obs = model._normalize(obs_batch)
        z_batch = torch.from_numpy(z[start:end]).to(device)
        logit = model._discriminator.compute_logits(normalized_obs, z_batch)
        logits.append(logit.float().squeeze(-1).cpu().numpy())
    logit_array = np.concatenate(logits).astype(np.float64, copy=False)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit_array, -80.0, 80.0)))
    return logit_array, probability


@torch.no_grad()
def _evaluate_mean_normalized_state(
    model: torch.nn.Module,
    z: np.ndarray,
    *,
    batch_size: int,
    device: str,
    state_dim: int,
    privileged_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        count = end - start
        normalized_obs = {
            "state": torch.zeros((count, state_dim), device=device),
            "privileged_state": torch.zeros((count, privileged_dim), device=device),
        }
        z_batch = torch.from_numpy(z[start:end]).to(device)
        logit = model._discriminator.compute_logits(normalized_obs, z_batch)
        logits.append(logit.float().squeeze(-1).cpu().numpy())
    logit_array = np.concatenate(logits).astype(np.float64, copy=False)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit_array, -80.0, 80.0)))
    return logit_array, probability


def _cell_stats(name: str, logits: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "count": int(logits.size),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "logit_median": float(np.median(logits)),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "probability_p05": float(np.percentile(probability, 5)),
        "probability_median": float(np.median(probability)),
        "probability_p95": float(np.percentile(probability, 95)),
        "predicted_expert_rate_at_0.5": float(np.mean(probability >= 0.5)),
    }


def _cosine_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1.0e-12)
    return {
        "mean": float(cosine.mean()),
        "std": float(cosine.std()),
        "p05": float(np.percentile(cosine, 5)),
        "median": float(np.median(cosine)),
        "p95": float(np.percentile(cosine, 95)),
    }


def _roc_auc(expert: np.ndarray, policy: np.ndarray) -> float:
    scores = np.concatenate([expert, policy])
    labels = np.concatenate(
        [np.ones(expert.size, dtype=bool), np.zeros(policy.size, dtype=bool)]
    )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    stops = np.r_[starts[1:], scores.size]
    for start, stop in zip(starts, stops):
        ranks[start:stop] = 0.5 * ((start + 1) + stop)
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[labels].sum()
    return float(
        (positive_rank_sum - expert.size * (expert.size + 1) / 2)
        / (expert.size * policy.size)
    )


def _threshold_separation(expert: np.ndarray, policy: np.ndarray) -> dict[str, float]:
    thresholds = np.linspace(0.0, 1.0, 10001)
    expert_sorted = np.sort(expert)
    policy_sorted = np.sort(policy)
    true_positive = 1.0 - np.searchsorted(
        expert_sorted, thresholds, side="left"
    ) / expert.size
    true_negative = np.searchsorted(
        policy_sorted, thresholds, side="left"
    ) / policy.size
    balanced_accuracy = 0.5 * (true_positive + true_negative)
    best = int(np.argmax(balanced_accuracy))
    return {
        "roc_auc": _roc_auc(expert, policy),
        "balanced_accuracy_at_0.5": float(
            0.5 * (np.mean(expert >= 0.5) + np.mean(policy < 0.5))
        ),
        "best_balanced_accuracy": float(balanced_accuracy[best]),
        "best_threshold": float(thresholds[best]),
    }


def _plot_cells(cells: dict[str, dict[str, Any]], output_path: Path) -> None:
    names = [
        "matched_expert",
        "matched_policy_stored_z",
        "policy_state_recomputed_z",
        "independent_expert_state_expert_z",
        "independent_expert_state_policy_z",
        "independent_policy_state_expert_z",
        "independent_policy_state_policy_z",
        "expert_state_common_random_z",
        "policy_state_common_random_z",
    ]
    labels = [
        "E state\nmatched E z",
        "P state\nstored P z",
        "P state\nrecomputed z",
        "E state\nshuffled E z",
        "E state\nshuffled P z",
        "P state\nshuffled E z",
        "P state\nshuffled P z",
        "E state\ncommon random z",
        "P state\ncommon random z",
    ]
    values = [cells[name]["logit_mean"] for name in names]
    colors = ["#D95319" if "expert" in name else "#2878B5" for name in names]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(np.arange(len(names)), values, color=colors, alpha=0.82)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
    ax.set_ylabel("Mean discriminator logit")
    ax.set_title("Training-free discriminator state/z counterfactuals")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> Path:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    data_path = args.data_path.expanduser().resolve()
    train_status = _read_json(checkpoint_dir / "train_status.json")
    global_time = int(train_status.get("global_time", train_status.get("time", 0)))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder / f"discriminator_counterfactuals_latest_{global_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")

    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model_from_checkpoint_dir(
        checkpoint_dir, device=checkpoint_load_device(args.device)
    )
    model.to(args.device)
    model.eval()
    seq_length = int(model.cfg.seq_length)

    buffer_paths = sorted(
        (checkpoint_dir / "buffers").glob("train_rank_*/buffer.hdf5"),
        key=lambda path: int(path.parent.name.rsplit("_", 1)[-1]),
    )
    policy_obs, policy_z, recomputed_policy_z, rank_metadata = _sample_policy_windows(
        model,
        buffer_paths,
        samples_per_rank=args.samples_per_rank,
        seq_length=seq_length,
        device=args.device,
        rng=rng,
    )
    num_samples = int(policy_z.shape[0])
    if num_samples % seq_length != 0:
        raise RuntimeError("Total sample count must be divisible by sequence length")

    env_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    try:
        expert_buffer = load_expert_trajectories_from_motion_lib(
            wrapped_env._env,
            SimpleNamespace(model=model.cfg),
            device=args.device,
        )
        expert_obs, expert_z = _sample_expert(
            model,
            expert_buffer,
            num_samples=num_samples,
            seq_length=seq_length,
        )
    finally:
        wrapped_env.close()

    permutation_expert = rng.permutation(num_samples)
    permutation_policy = rng.permutation(num_samples)
    expert_z_independent = expert_z[permutation_expert]
    policy_z_independent = policy_z[permutation_policy]
    recomputed_z_independent = recomputed_policy_z[rng.permutation(num_samples)]
    random_z_tensor = model.project_z(
        torch.randn((num_samples, expert_z.shape[1]), device=args.device)
    )
    common_random_z = random_z_tensor.float().cpu().numpy()

    pair_specs = {
        "matched_expert": (expert_obs, expert_z),
        "matched_policy_stored_z": (policy_obs, policy_z),
        "policy_state_recomputed_z": (policy_obs, recomputed_policy_z),
        "independent_expert_state_expert_z": (expert_obs, expert_z_independent),
        "independent_expert_state_policy_z": (expert_obs, policy_z_independent),
        "independent_policy_state_expert_z": (policy_obs, expert_z_independent),
        "independent_policy_state_policy_z": (policy_obs, policy_z_independent),
        "independent_expert_state_recomputed_policy_z": (
            expert_obs,
            recomputed_z_independent,
        ),
        "independent_policy_state_recomputed_policy_z": (
            policy_obs,
            recomputed_z_independent,
        ),
        "expert_state_common_random_z": (expert_obs, common_random_z),
        "policy_state_common_random_z": (policy_obs, common_random_z),
    }
    cells: dict[str, dict[str, Any]] = {}
    raw: dict[str, np.ndarray] = {}
    for name, (obs, z) in pair_specs.items():
        logits, probability = _evaluate_pair(
            model,
            obs,
            z,
            batch_size=args.batch_size,
            device=args.device,
        )
        cells[name] = _cell_stats(name, logits, probability)
        raw[f"{name}_logit"] = logits.astype(np.float32)
        raw[f"{name}_probability"] = probability.astype(np.float32)
        print(
            f"[CF-DISC] {name}: logit={logits.mean():.4f} "
            f"prob={probability.mean():.4f} expert_rate={np.mean(probability >= 0.5):.4f}",
            flush=True,
        )

    state_dim = expert_obs["state"].shape[-1]
    privileged_dim = expert_obs["privileged_state"].shape[-1]
    for name, z in {
        "mean_normalized_state_expert_z": expert_z_independent,
        "mean_normalized_state_policy_z": policy_z_independent,
        "mean_normalized_state_recomputed_policy_z": recomputed_z_independent,
    }.items():
        logits, probability = _evaluate_mean_normalized_state(
            model,
            z,
            batch_size=args.batch_size,
            device=args.device,
            state_dim=state_dim,
            privileged_dim=privileged_dim,
        )
        cells[name] = _cell_stats(name, logits, probability)
        raw[f"{name}_logit"] = logits.astype(np.float32)
        raw[f"{name}_probability"] = probability.astype(np.float32)

    ee = cells["independent_expert_state_expert_z"]["logit_mean"]
    ep = cells["independent_expert_state_policy_z"]["logit_mean"]
    pe = cells["independent_policy_state_expert_z"]["logit_mean"]
    pp = cells["independent_policy_state_policy_z"]["logit_mean"]
    state_main_effect = 0.5 * ((ee - pe) + (ep - pp))
    z_main_effect = 0.5 * ((ee - ep) + (pe - pp))
    interaction = (ee - ep) - (pe - pp)
    denominator = abs(state_main_effect) + abs(z_main_effect)

    common_random_state_accuracy = 0.5 * (
        cells["expert_state_common_random_z"]["predicted_expert_rate_at_0.5"]
        + 1.0
        - cells["policy_state_common_random_z"]["predicted_expert_rate_at_0.5"]
    )
    matched_accuracy = 0.5 * (
        cells["matched_expert"]["predicted_expert_rate_at_0.5"]
        + 1.0
        - cells["matched_policy_stored_z"]["predicted_expert_rate_at_0.5"]
    )
    effects = {
        "matched_balanced_accuracy": float(matched_accuracy),
        "factorial_state_main_effect_logit": float(state_main_effect),
        "factorial_z_source_main_effect_logit": float(z_main_effect),
        "factorial_state_z_interaction_logit": float(interaction),
        "z_source_absolute_effect_fraction": float(
            abs(z_main_effect) / denominator if denominator else 0.0
        ),
        "common_random_z_state_balanced_accuracy": float(
            common_random_state_accuracy
        ),
        "policy_probability_change_recomputed_minus_stored_z": float(
            cells["policy_state_recomputed_z"]["probability_mean"]
            - cells["matched_policy_stored_z"]["probability_mean"]
        ),
        "policy_logit_change_recomputed_minus_stored_z": float(
            cells["policy_state_recomputed_z"]["logit_mean"]
            - cells["matched_policy_stored_z"]["logit_mean"]
        ),
        "expert_matched_state_z_compatibility_logit_gain": float(
            cells["matched_expert"]["logit_mean"]
            - cells["independent_expert_state_expert_z"]["logit_mean"]
        ),
        "policy_matched_state_z_compatibility_logit_gain": float(
            cells["matched_policy_stored_z"]["logit_mean"]
            - cells["independent_policy_state_policy_z"]["logit_mean"]
        ),
        "policy_stored_vs_recomputed_z_cosine": _cosine_stats(
            policy_z, recomputed_policy_z
        ),
        "z_norms": {
            "expert_mean": float(np.linalg.norm(expert_z, axis=1).mean()),
            "policy_stored_mean": float(np.linalg.norm(policy_z, axis=1).mean()),
            "policy_recomputed_mean": float(
                np.linalg.norm(recomputed_policy_z, axis=1).mean()
            ),
        },
    }
    fixed_z_separation = {
        "matched_baseline": _threshold_separation(
            raw["matched_expert_probability"],
            raw["matched_policy_stored_z_probability"],
        ),
        "same_expert_z_for_both_state_sources": _threshold_separation(
            raw["independent_expert_state_expert_z_probability"],
            raw["independent_policy_state_expert_z_probability"],
        ),
        "same_policy_z_for_both_state_sources": _threshold_separation(
            raw["independent_expert_state_policy_z_probability"],
            raw["independent_policy_state_policy_z_probability"],
        ),
        "same_recomputed_policy_z_for_both_state_sources": _threshold_separation(
            raw["independent_expert_state_recomputed_policy_z_probability"],
            raw["independent_policy_state_recomputed_policy_z_probability"],
        ),
        "same_common_random_z_for_both_state_sources": _threshold_separation(
            raw["expert_state_common_random_z_probability"],
            raw["policy_state_common_random_z_probability"],
        ),
        "z_source_with_mean_normalized_state": _threshold_separation(
            raw["mean_normalized_state_expert_z_probability"],
            raw["mean_normalized_state_policy_z_probability"],
        ),
    }

    cells_csv = output_dir / "counterfactual_cells.csv"
    pd.DataFrame(cells.values()).to_csv(cells_csv, index=False)
    raw_path = output_dir / "counterfactual_outputs.npz"
    np.savez_compressed(raw_path, **raw)
    figure_path = output_dir / "counterfactual_logits.png"
    _plot_cells(cells, figure_path)
    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_global_time": global_time,
        "expert_dataset": str(data_path),
        "num_samples_per_class": num_samples,
        "sequence_length": seq_length,
        "method": {
            "no_training": True,
            "factorial_cells_use_independently_shuffled_z": True,
            "policy_recomputed_z": "project(mean(backward(next 8 policy states)))",
            "logit_used_for_effect_sizes": True,
        },
        "effects": effects,
        "fixed_z_separation": fixed_z_separation,
        "cells": cells,
        "policy_sampling": rank_metadata,
        "files": {
            "cells_csv": str(cells_csv),
            "raw_outputs": str(raw_path),
            "plot": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(effects), flush=True)
    print(f"[CF-DISC] wrote {summary_path}", flush=True)
    return output_dir


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
