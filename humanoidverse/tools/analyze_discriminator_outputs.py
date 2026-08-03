"""Analyze latest-checkpoint discriminator probabilities on expert and policy data."""

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
        description="Compare sigmoid(discriminator) for expert and policy replay states."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-time-rows-per-rank", type=int, default=32)
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
def _policy_probabilities(
    model: torch.nn.Module,
    buffer_paths: list[Path],
    *,
    rows_per_rank: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_probabilities: list[np.ndarray] = []
    per_rank: list[dict[str, Any]] = []
    required = ("observation-state", "observation-privileged_state", "z")

    for rank, path in enumerate(buffer_paths):
        with h5py.File(path, "r") as file:
            missing = [key for key in required if key not in file]
            if missing:
                raise KeyError(f"Missing datasets in {path}: {missing}")
            num_time_rows, num_envs = file[required[0]].shape[:2]
            if rows_per_rank > num_time_rows:
                raise ValueError(
                    f"Requested {rows_per_rank} rows from rank {rank}, but only {num_time_rows} exist"
                )
            row_indices = np.sort(
                rng.choice(num_time_rows, size=rows_per_rank, replace=False)
            )
            state = file[required[0]][row_indices].reshape(-1, file[required[0]].shape[-1])
            privileged = file[required[1]][row_indices].reshape(
                -1, file[required[1]].shape[-1]
            )
            z = file[required[2]][row_indices].reshape(-1, file[required[2]].shape[-1])

        rank_probabilities: list[np.ndarray] = []
        for start in range(0, state.shape[0], batch_size):
            end = min(start + batch_size, state.shape[0])
            obs = {
                "state": torch.from_numpy(state[start:end]).to(device),
                "privileged_state": torch.from_numpy(privileged[start:end]).to(device),
            }
            z_batch = torch.from_numpy(z[start:end]).to(device)
            probabilities = model.discriminator(obs, z_batch).float().squeeze(-1)
            rank_probabilities.append(probabilities.cpu().numpy())

        rank_array = np.concatenate(rank_probabilities).astype(np.float64, copy=False)
        all_probabilities.append(rank_array)
        per_rank.append(
            {
                "rank": rank,
                "buffer": str(path),
                "num_time_rows_in_buffer": int(num_time_rows),
                "num_envs": int(num_envs),
                "sampled_time_rows": row_indices.tolist(),
                "num_samples": int(rank_array.size),
                "mean": float(rank_array.mean()),
                "std": float(rank_array.std()),
                "predicted_policy_rate_at_0.5": float(np.mean(rank_array < 0.5)),
            }
        )
        print(
            f"[DISC] policy rank={rank} samples={rank_array.size} "
            f"mean={rank_array.mean():.6f} policy_correct={np.mean(rank_array < 0.5):.4f}",
            flush=True,
        )

    return np.concatenate(all_probabilities), per_rank


@torch.no_grad()
def _expert_probabilities(
    model: torch.nn.Module,
    expert_buffer: Any,
    *,
    num_samples: int,
    batch_size: int,
) -> np.ndarray:
    seq_length = int(model.cfg.seq_length)
    if batch_size % seq_length != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by seq_length={seq_length}"
        )
    probabilities: list[np.ndarray] = []
    collected = 0
    while collected < num_samples:
        current = min(batch_size, num_samples - collected)
        current -= current % seq_length
        if current == 0:
            raise ValueError("num_samples must be divisible by the model sequence length")
        batch = expert_buffer.sample(current, seq_length=seq_length)
        obs = batch["observation"]
        next_obs = batch["next"]["observation"]

        backward = model.backward_map(next_obs)
        expert_z = backward.view(
            current // seq_length, seq_length, backward.shape[-1]
        ).mean(dim=1)
        expert_z = model.project_z(expert_z)
        expert_z = torch.repeat_interleave(expert_z, seq_length, dim=0)
        probability = model.discriminator(obs, expert_z).float().squeeze(-1)
        probabilities.append(probability.cpu().numpy())
        collected += current

    return np.concatenate(probabilities).astype(np.float64, copy=False)


def _describe(probabilities: np.ndarray, *, positive: bool) -> dict[str, float | int]:
    correct = probabilities >= 0.5 if positive else probabilities < 0.5
    eps = 1.0e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    bce = -np.log(clipped) if positive else -np.log(1.0 - clipped)
    return {
        "count": int(probabilities.size),
        "mean": float(probabilities.mean()),
        "std": float(probabilities.std()),
        "min": float(probabilities.min()),
        "p01": float(np.percentile(probabilities, 1)),
        "p05": float(np.percentile(probabilities, 5)),
        "p10": float(np.percentile(probabilities, 10)),
        "p25": float(np.percentile(probabilities, 25)),
        "median": float(np.median(probabilities)),
        "p75": float(np.percentile(probabilities, 75)),
        "p90": float(np.percentile(probabilities, 90)),
        "p95": float(np.percentile(probabilities, 95)),
        "p99": float(np.percentile(probabilities, 99)),
        "max": float(probabilities.max()),
        "classification_accuracy_at_0.5": float(correct.mean()),
        "ambiguous_rate_0.4_to_0.6": float(
            np.mean((probabilities >= 0.4) & (probabilities <= 0.6))
        ),
        "rate_below_0.1": float(np.mean(probabilities < 0.1)),
        "rate_above_0.9": float(np.mean(probabilities > 0.9)),
        "bce_mean": float(bce.mean()),
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


def _threshold_metrics(expert: np.ndarray, policy: np.ndarray) -> dict[str, float]:
    thresholds = np.linspace(0.0, 1.0, 10001)
    expert_sorted = np.sort(expert)
    policy_sorted = np.sort(policy)
    expert_true_positive = 1.0 - np.searchsorted(
        expert_sorted, thresholds, side="left"
    ) / expert.size
    policy_true_negative = np.searchsorted(
        policy_sorted, thresholds, side="left"
    ) / policy.size
    balanced_accuracy = 0.5 * (expert_true_positive + policy_true_negative)
    best = int(np.argmax(balanced_accuracy))
    return {
        "balanced_accuracy_at_0.5": float(
            0.5 * (np.mean(expert >= 0.5) + np.mean(policy < 0.5))
        ),
        "best_balanced_accuracy": float(balanced_accuracy[best]),
        "best_threshold": float(thresholds[best]),
        "expert_true_positive_rate_at_best_threshold": float(expert_true_positive[best]),
        "policy_true_negative_rate_at_best_threshold": float(policy_true_negative[best]),
    }


def _write_histogram(
    expert: np.ndarray, policy: np.ndarray, output_dir: Path
) -> tuple[Path, Path, float]:
    edges = np.linspace(0.0, 1.0, 101)
    expert_hist, _ = np.histogram(expert, bins=edges)
    policy_hist, _ = np.histogram(policy, bins=edges)
    expert_mass = expert_hist / expert_hist.sum()
    policy_mass = policy_hist / policy_hist.sum()
    overlap = float(np.minimum(expert_mass, policy_mass).sum())

    histogram_csv = output_dir / "sigmoid_histogram.csv"
    pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "expert_count": expert_hist,
            "expert_probability_mass": expert_mass,
            "policy_count": policy_hist,
            "policy_probability_mass": policy_mass,
        }
    ).to_csv(histogram_csv, index=False)

    figure_path = output_dir / "sigmoid_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        policy,
        bins=edges,
        density=True,
        alpha=0.58,
        color="#2878B5",
        label=f"Policy replay (mean={policy.mean():.3f})",
    )
    ax.hist(
        expert,
        bins=edges,
        density=True,
        alpha=0.58,
        color="#D95319",
        label=f"Expert (mean={expert.mean():.3f})",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="threshold=0.5")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("sigmoid(discriminator logit): P(expert | state, z)")
    ax.set_ylabel("Density")
    ax.set_title("Latest-checkpoint discriminator outputs")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return histogram_csv, figure_path, overlap


def analyze(args: argparse.Namespace) -> Path:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    data_path = args.data_path.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing expert dataset: {data_path}")

    train_status = _read_json(checkpoint_dir / "train_status.json")
    global_time = int(train_status.get("global_time", train_status.get("time", 0)))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder / f"discriminator_analysis_latest_{global_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {summary_path}")

    buffer_paths = sorted(
        (checkpoint_dir / "buffers").glob("train_rank_*/buffer.hdf5"),
        key=lambda path: int(path.parent.name.rsplit("_", 1)[-1]),
    )
    if not buffer_paths:
        raise FileNotFoundError("No rank-local replay buffers found")

    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model_from_checkpoint_dir(
        checkpoint_dir, device=checkpoint_load_device(args.device)
    )
    model.to(args.device)
    model.eval()

    policy, per_rank = _policy_probabilities(
        model,
        buffer_paths,
        rows_per_rank=args.policy_time_rows_per_rank,
        batch_size=args.batch_size,
        device=args.device,
        rng=rng,
    )

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
        expert = _expert_probabilities(
            model,
            expert_buffer,
            num_samples=policy.size,
            batch_size=args.batch_size,
        )
    finally:
        wrapped_env.close()

    histogram_csv, figure_path, overlap = _write_histogram(expert, policy, output_dir)
    samples_path = output_dir / "sampled_probabilities.npz"
    np.savez_compressed(samples_path, expert=expert, policy=policy)

    expert_stats = _describe(expert, positive=True)
    policy_stats = _describe(policy, positive=False)
    separation = {
        "roc_auc_expert_positive": _roc_auc(expert, policy),
        "histogram_overlap_coefficient_100_bins": overlap,
        "mean_gap_expert_minus_policy": float(expert.mean() - policy.mean()),
        **_threshold_metrics(expert, policy),
    }
    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_global_time": global_time,
        "checkpoint_optimizer_steps": int(train_status.get("optimizer_steps", 0)),
        "expert_dataset": str(data_path),
        "interpretation": "sigmoid output is P(expert | state, privileged_state, z)",
        "sampling": {
            "seed": int(args.seed),
            "policy_source": "all rank-local replay buffers from the latest checkpoint",
            "policy_time_rows_per_rank": int(args.policy_time_rows_per_rank),
            "num_policy_samples": int(policy.size),
            "num_expert_samples": int(expert.size),
            "expert_motion_sampling": "uniform over motions, training-style contiguous sequences",
            "sequence_length": int(model.cfg.seq_length),
        },
        "expert_positive": expert_stats,
        "policy_negative": policy_stats,
        "separation": separation,
        "policy_per_rank": per_rank,
        "files": {
            "histogram_csv": str(histogram_csv),
            "distribution_plot": str(figure_path),
            "sampled_probabilities": str(samples_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "expert": expert_stats,
                "policy": policy_stats,
                "separation": separation,
            }
        ),
        flush=True,
    )
    print(f"[DISC] wrote {summary_path}", flush=True)
    print(f"[DISC] wrote {figure_path}", flush=True)
    return output_dir


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
