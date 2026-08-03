"""Test whether the discriminator separates the minimum-EMD rollout from its expert target."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import emd_numpy, get_backward_observation
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.tracking_inference import _target_states_from_obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out the minimum-EMD motion and compare discriminator outputs with its expert target."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--emd-column", choices=["emd", "obs_state_emd"], default="emd")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_minimum_emd(
    eval_csv: Path, *, checkpoint_global_time: int, emd_column: str
) -> dict[str, Any]:
    frame = pd.read_csv(eval_csv)
    required = {"timestep", "motion_id", "motion_file", emd_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in {eval_csv}: {sorted(missing)}")
    eligible = frame[frame["timestep"] <= checkpoint_global_time]
    if eligible.empty:
        raise ValueError(
            f"No evaluation timestep <= checkpoint global_time={checkpoint_global_time}"
        )
    timestep = int(eligible["timestep"].max())
    row = eligible[eligible["timestep"] == timestep].nsmallest(1, emd_column).iloc[0]
    return {
        "evaluation_timestep": timestep,
        "motion_id_in_full_dataset": int(row["motion_id"]),
        "motion_file": str(row["motion_file"]),
        "selection_metric": emd_column,
        "selection_emd": float(row[emd_column]),
        "recorded_joint_position_emd": float(row["emd"]),
        "recorded_observation_state_emd": float(row["obs_state_emd"]),
    }


def _write_single_motion_dataset(data_path: Path, motion_file: str, output_path: Path) -> None:
    dataset = joblib.load(data_path, mmap_mode="r")
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dict dataset, got {type(dataset)}")
    if motion_file not in dataset:
        raise KeyError(f"Motion {motion_file!r} not found in {data_path}")
    joblib.dump({motion_file: dataset[motion_file]}, output_path, compress=0)


def _future_mean_z(
    model: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    *,
    sequence_length: int,
) -> torch.Tensor:
    encoded = model.backward_map(obs)
    if encoded.shape[0] < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} future states, got {encoded.shape[0]}"
        )
    windows = encoded.unfold(0, sequence_length, 1).mean(dim=-1)
    return model.project_z(windows)


@torch.no_grad()
def _probabilities(
    model: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    z: torch.Tensor,
) -> np.ndarray:
    count = min(z.shape[0], *(value.shape[0] for value in obs.values()))
    probability = model.discriminator(
        {key: value[:count] for key, value in obs.items()}, z[:count]
    )
    return probability.float().squeeze(-1).cpu().numpy().astype(np.float64, copy=False)


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


def _pair_stats(expert: np.ndarray, policy: np.ndarray) -> dict[str, Any]:
    count = min(expert.size, policy.size)
    expert = expert[:count]
    policy = policy[:count]
    thresholds = np.linspace(0.0, 1.0, 10001)
    expert_sorted = np.sort(expert)
    policy_sorted = np.sort(policy)
    true_positive = 1.0 - np.searchsorted(expert_sorted, thresholds, side="left") / count
    true_negative = np.searchsorted(policy_sorted, thresholds, side="left") / count
    balanced = 0.5 * (true_positive + true_negative)
    best = int(np.argmax(balanced))
    expert_hist, _ = np.histogram(expert, bins=np.linspace(0.0, 1.0, 101))
    policy_hist, _ = np.histogram(policy, bins=np.linspace(0.0, 1.0, 101))
    overlap = np.minimum(expert_hist / count, policy_hist / count).sum()
    return {
        "count_per_class": int(count),
        "expert_probability": {
            "mean": float(expert.mean()),
            "std": float(expert.std()),
            "p05": float(np.percentile(expert, 5)),
            "median": float(np.median(expert)),
            "p95": float(np.percentile(expert, 95)),
            "predicted_expert_rate_at_0.5": float(np.mean(expert >= 0.5)),
        },
        "policy_probability": {
            "mean": float(policy.mean()),
            "std": float(policy.std()),
            "p05": float(np.percentile(policy, 5)),
            "median": float(np.median(policy)),
            "p95": float(np.percentile(policy, 95)),
            "predicted_policy_rate_at_0.5": float(np.mean(policy < 0.5)),
        },
        "balanced_accuracy_at_0.5": float(
            0.5 * (np.mean(expert >= 0.5) + np.mean(policy < 0.5))
        ),
        "roc_auc": _roc_auc(expert, policy),
        "best_balanced_accuracy": float(balanced[best]),
        "best_threshold": float(thresholds[best]),
        "histogram_overlap": float(overlap),
        "paired_expert_probability_greater_rate": float(np.mean(expert > policy)),
        "paired_probability_gap_mean": float(np.mean(expert - policy)),
    }


def evaluate(args: argparse.Namespace) -> Path:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    data_path = args.data_path.expanduser().resolve()
    eval_csv = (
        args.eval_csv.expanduser().resolve()
        if args.eval_csv is not None
        else model_folder / "humanoidverse_tracking_eval.csv"
    )
    train_status = _read_json(checkpoint_dir / "train_status.json")
    global_time = int(train_status.get("global_time", train_status.get("time", 0)))
    selection = _select_minimum_emd(
        eval_csv,
        checkpoint_global_time=global_time,
        emd_column=args.emd_column,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder
        / f"min_emd_discriminator_{global_time}_{selection['motion_id_in_full_dataset']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")

    subset_path = output_dir / "selected_motion.pkl"
    _write_single_motion_dataset(data_path, selection["motion_file"], subset_path)
    _set_seed(args.seed)
    print(
        "[MIN-EMD-DISC] "
        f"checkpoint={global_time} eval_timestep={selection['evaluation_timestep']} "
        f"motion_id={selection['motion_id_in_full_dataset']} "
        f"motion={selection['motion_file']} recorded_emd={selection['selection_emd']:.6f}",
        flush=True,
    )

    model = load_model_from_checkpoint_dir(
        checkpoint_dir, device=checkpoint_load_device(args.device)
    )
    model.to(args.device)
    model.eval()
    env_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=subset_path,
        robot_config=None,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=10000.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env.is_evaluating = True

    try:
        expert_obs_all, reference = get_backward_observation(
            env, 0, include_last_action=True
        )
        expert_obs_all = {
            key: value.to(args.device)
            for key, value in expert_obs_all.items()
            if key in {"state", "privileged_state"}
        }
        target_single_z = model.project_z(
            model.backward_map({key: value[1:] for key, value in expert_obs_all.items()})
        )
        target_states = _target_states_from_obs(
            reference, device=args.device, num_dof=env.num_dof
        )
        observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)

        policy_state: list[torch.Tensor] = []
        policy_privileged: list[torch.Tensor] = []
        joint_positions = [env.simulator.dof_state[..., 0].detach().clone()]
        ended_early = False
        for step in range(target_single_z.shape[0]):
            policy_state.append(observation["state"].detach().clone())
            policy_privileged.append(observation["privileged_state"].detach().clone())
            action = model.act(observation, target_single_z[step : step + 1], mean=True)
            observation, _reward, terminated, truncated, _info = wrapped_env.step(
                action, to_numpy=False
            )
            joint_positions.append(env.simulator.dof_state[..., 0].detach().clone())
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                ended_early = True
                break

        policy_obs_all = {
            "state": torch.cat(policy_state, dim=0),
            "privileged_state": torch.cat(policy_privileged, dim=0),
        }
        rollout_steps = policy_obs_all["state"].shape[0]
        expert_obs = {key: value[:rollout_steps] for key, value in expert_obs_all.items()}
        target_single_z = target_single_z[:rollout_steps]

        sequence_length = int(model.cfg.seq_length)
        common_length = rollout_steps - sequence_length
        target_future_z = _future_mean_z(
            model,
            {key: value[1 : common_length + sequence_length + 1] for key, value in expert_obs_all.items()},
            sequence_length=sequence_length,
        )[:common_length]
        rollout_future_z = _future_mean_z(
            model,
            {key: value[1:] for key, value in policy_obs_all.items()},
            sequence_length=sequence_length,
        )[:common_length]
        expert_common = {key: value[:common_length] for key, value in expert_obs.items()}
        policy_common = {key: value[:common_length] for key, value in policy_obs_all.items()}

        pair_inputs = {
            "same_target_single_frame_z": (
                expert_obs,
                policy_obs_all,
                target_single_z,
                target_single_z,
            ),
            "same_target_future_8frame_z": (
                expert_common,
                policy_common,
                target_future_z,
                target_future_z,
            ),
            "same_rollout_future_8frame_z": (
                expert_common,
                policy_common,
                rollout_future_z,
                rollout_future_z,
            ),
            "natural_source_future_8frame_z": (
                expert_common,
                policy_common,
                target_future_z,
                rollout_future_z,
            ),
        }
        pairs: dict[str, Any] = {}
        frame_output: dict[str, np.ndarray] = {}
        for name, (expert_pair_obs, policy_pair_obs, expert_z, policy_z) in pair_inputs.items():
            expert_probability = _probabilities(model, expert_pair_obs, expert_z)
            policy_probability = _probabilities(model, policy_pair_obs, policy_z)
            pairs[name] = _pair_stats(expert_probability, policy_probability)
            frame_output[f"{name}_expert_probability"] = expert_probability
            frame_output[f"{name}_policy_probability"] = policy_probability
            print(
                f"[MIN-EMD-DISC] {name}: "
                f"expert={expert_probability.mean():.6f} policy={policy_probability.mean():.6f} "
                f"auc={pairs[name]['roc_auc']:.6f} "
                f"balanced_acc@0.5={pairs[name]['balanced_accuracy_at_0.5']:.6f}",
                flush=True,
            )

        joint_position = torch.cat(joint_positions, dim=0)
        target_joint_position = reference["dof_pos"][: joint_position.shape[0]]
        reproduced_joint_emd = float(
            emd_numpy(joint_position, target_joint_position)["emd"]
        )
        reproduced_obs_state_emd = float(
            emd_numpy(
                policy_obs_all["state"][:, :23], expert_obs["state"][:, :23]
            )["emd"]
        )
    finally:
        wrapped_env.close()

    np.savez_compressed(output_dir / "per_frame_probabilities.npz", **frame_output)
    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_global_time": global_time,
        "expert_dataset": str(data_path),
        "evaluation_csv": str(eval_csv),
        "selection": selection,
        "controlled_rollout": {
            "seed": int(args.seed),
            "domain_randomization_disabled": True,
            "observation_noise_disabled": True,
            "rollout_steps": int(rollout_steps),
            "ended_early": bool(ended_early),
            "reproduced_joint_position_emd": reproduced_joint_emd,
            "reproduced_observation_state_emd": reproduced_obs_state_emd,
        },
        "interpretation": {
            "sigmoid_semantics": "P(expert | state, z)",
            "expert_is_positive": True,
            "policy_is_negative": True,
            "same_z_pairs_isolate_state_source": True,
            "sequence_length": int(model.cfg.seq_length),
        },
        "pairs": pairs,
        "files": {
            "selected_motion_dataset": str(subset_path),
            "per_frame_probabilities": str(output_dir / "per_frame_probabilities.npz"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[MIN-EMD-DISC] wrote {summary_path}", flush=True)
    return output_dir


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
