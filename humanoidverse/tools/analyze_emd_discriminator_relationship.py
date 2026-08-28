"""Measure how tracking EMD relates to discriminator separability.

This is a training-free diagnostic.  It selects representative motions from
fixed EMD ranges, reproduces their deterministic tracking rollouts, and scores
expert/policy frames under both training-style and controlled shared-z inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    emd_numpy,
    get_backward_observation,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import (
    checkpoint_load_device,
    load_mjlab_env_cfg,
)
from humanoidverse.tools.eval_min_emd_discriminator import (
    _future_mean_z,
    _pair_stats,
    _probabilities,
)
from humanoidverse.tracking_inference import _target_states_from_obs


DEFAULT_BIN_EDGES = "0,0.3,0.35,0.4,0.45,0.5,0.6,0.75,1,1.5,2,inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate per-motion tracking EMD to discriminator outputs."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-bin", type=int, default=5)
    parser.add_argument("--bin-edges", default=DEFAULT_BIN_EDGES)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Reuse a complete per_motion.csv and only regenerate aggregate outputs.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_bin_edges(value: str) -> np.ndarray:
    edges = np.asarray(
        [float("inf") if token.strip().lower() == "inf" else float(token)
         for token in value.split(",")],
        dtype=np.float64,
    )
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError(f"Bin edges must be strictly increasing, got {value!r}")
    return edges


def _latest_evaluation_rows(eval_csv: Path, checkpoint_global_time: int) -> tuple[int, pd.DataFrame]:
    frame = pd.read_csv(eval_csv)
    required = {"timestep", "motion_id", "motion_file", "emd", "obs_state_emd"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in {eval_csv}: {sorted(missing)}")
    eligible = frame[frame["timestep"] <= checkpoint_global_time]
    if eligible.empty:
        raise ValueError(
            f"No evaluation timestep <= checkpoint global_time={checkpoint_global_time}"
        )
    timestep = int(eligible["timestep"].max())
    rows = eligible[eligible["timestep"] == timestep].copy()
    if rows["motion_id"].duplicated().any():
        raise ValueError(f"Duplicate motion IDs at evaluation timestep {timestep}")
    return timestep, rows.sort_values("motion_id").reset_index(drop=True)


def _select_representative_rows(
    rows: pd.DataFrame, edges: np.ndarray, samples_per_bin: int
) -> pd.DataFrame:
    if samples_per_bin <= 0:
        raise ValueError("samples_per_bin must be positive")
    selected: list[pd.DataFrame] = []
    emd = rows["emd"].to_numpy(dtype=np.float64)
    for bin_id, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (emd >= low) & (emd < high)
        candidates = rows.loc[mask].sort_values(["emd", "motion_id"]).reset_index(drop=True)
        if candidates.empty:
            continue
        count = min(samples_per_bin, len(candidates))
        positions = np.floor((np.arange(count) + 0.5) * len(candidates) / count).astype(int)
        positions = np.clip(positions, 0, len(candidates) - 1)
        current = candidates.iloc[positions].copy()
        high_label = "inf" if math.isinf(high) else f"{high:g}"
        current["emd_bin_id"] = bin_id
        current["emd_bin"] = f"[{low:g},{high_label})"
        current["emd_bin_dataset_count"] = len(candidates)
        selected.append(current)
    if not selected:
        raise ValueError("No motions fall inside the requested EMD bins")
    result = pd.concat(selected, ignore_index=True)
    if result["motion_id"].duplicated().any():
        raise RuntimeError("Representative selection unexpectedly duplicated a motion")
    return result


def _write_subset_dataset(data_path: Path, rows: pd.DataFrame, output_path: Path) -> None:
    dataset = joblib.load(data_path, mmap_mode="r")
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dict dataset, got {type(dataset)}")
    keys = rows["motion_file"].astype(str).tolist()
    missing = [key for key in keys if key not in dataset]
    if missing:
        raise KeyError(f"Selected motions missing from dataset: {missing[:3]}")
    joblib.dump({key: dataset[key] for key in keys}, output_path, compress=0)
    del dataset
    gc.collect()


def _flatten_pair(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    expert = stats["expert_probability"]
    policy = stats["policy_probability"]
    return {
        f"{prefix}_expert_probability_mean": expert["mean"],
        f"{prefix}_policy_probability_mean": policy["mean"],
        f"{prefix}_policy_correct_rate_at_0.5": policy["predicted_policy_rate_at_0.5"],
        f"{prefix}_policy_fooled_rate_at_0.5": 1.0 - policy["predicted_policy_rate_at_0.5"],
        f"{prefix}_balanced_accuracy_at_0.5": stats["balanced_accuracy_at_0.5"],
        f"{prefix}_roc_auc": stats["roc_auc"],
        f"{prefix}_best_balanced_accuracy": stats["best_balanced_accuracy"],
        f"{prefix}_best_threshold": stats["best_threshold"],
        f"{prefix}_histogram_overlap": stats["histogram_overlap"],
        f"{prefix}_probability_gap_mean": stats["paired_probability_gap_mean"],
    }


@torch.no_grad()
def _evaluate_motion(
    *,
    wrapped_env: Any,
    model: torch.nn.Module,
    local_motion_id: int,
    device: str,
) -> dict[str, Any]:
    env = wrapped_env._env
    motion_id_tensor = torch.tensor([local_motion_id], device=env.device, dtype=torch.long)
    env.motion_ids.copy_(motion_id_tensor)
    env.motion_len.copy_(env._motion_lib.get_motion_length(motion_id_tensor))
    env.motion_start_times.zero_()

    expert_obs_all, reference = get_backward_observation(
        env,
        local_motion_id,
        include_last_action=bool(wrapped_env.include_last_action),
    )
    expert_obs_all = {
        key: value.to(device)
        for key, value in expert_obs_all.items()
        if key in {"state", "privileged_state"}
    }
    target_single_z = model.project_z(
        model.backward_map({key: value[1:] for key, value in expert_obs_all.items()})
    )
    target_states = _target_states_from_obs(reference, device=device, num_dof=env.num_dof)
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
    rollout_steps = int(policy_obs_all["state"].shape[0])
    sequence_length = int(model.cfg.seq_length)
    common_length = rollout_steps - sequence_length
    if common_length <= 0:
        raise ValueError(
            f"Motion {local_motion_id} rollout has {rollout_steps} steps, "
            f"not enough for seq_length={sequence_length}"
        )

    expert_obs = {key: value[:rollout_steps] for key, value in expert_obs_all.items()}
    expert_common = {key: value[:common_length] for key, value in expert_obs.items()}
    policy_common = {key: value[:common_length] for key, value in policy_obs_all.items()}
    target_future_z = _future_mean_z(
        model,
        {
            key: value[1 : common_length + sequence_length + 1]
            for key, value in expert_obs_all.items()
        },
        sequence_length=sequence_length,
    )[:common_length]
    rollout_future_z = _future_mean_z(
        model,
        {key: value[1:] for key, value in policy_obs_all.items()},
        sequence_length=sequence_length,
    )[:common_length]

    pair_inputs = {
        # Closest approximation to D's training inputs for this guided rollout:
        # expert samples use the mean encoding of their next 8 expert states,
        # while policy samples retain the actual single-frame guidance z.
        "training_z": (
            expert_common,
            policy_common,
            target_future_z,
            target_single_z[:common_length],
        ),
        # Give both state sources exactly the policy's guidance z to isolate state.
        "shared_guidance_z": (
            expert_common,
            policy_common,
            target_single_z[:common_length],
            target_single_z[:common_length],
        ),
        # Give both sources the training-style expert future z to isolate state.
        "shared_expert_future_z": (
            expert_common,
            policy_common,
            target_future_z,
            target_future_z,
        ),
        # Counterfactual where each source's z is recomputed from its own future.
        "source_future_z": (
            expert_common,
            policy_common,
            target_future_z,
            rollout_future_z,
        ),
    }
    output: dict[str, Any] = {}
    for name, (expert_pair_obs, policy_pair_obs, expert_z, policy_z) in pair_inputs.items():
        stats = _pair_stats(
            _probabilities(model, expert_pair_obs, expert_z),
            _probabilities(model, policy_pair_obs, policy_z),
        )
        output.update(_flatten_pair(name, stats))

    joint_position = torch.cat(joint_positions, dim=0)
    target_joint_position = reference["dof_pos"][: joint_position.shape[0]]
    num_dof = int(env.num_dof)
    output.update(
        {
            "rollout_steps": rollout_steps,
            "ended_early": ended_early,
            "reproduced_emd": float(
                emd_numpy(joint_position, target_joint_position)["emd"]
            ),
            "reproduced_obs_state_emd": float(
                emd_numpy(
                    policy_obs_all["state"][:, :num_dof],
                    expert_obs["state"][:, :num_dof],
                )["emd"]
            ),
        }
    )
    return output


def _relationship_summary(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    columns = [
        f"{prefix}_policy_probability_mean",
        f"{prefix}_policy_correct_rate_at_0.5",
        f"{prefix}_policy_fooled_rate_at_0.5",
        f"{prefix}_balanced_accuracy_at_0.5",
        f"{prefix}_roc_auc",
        f"{prefix}_best_balanced_accuracy",
        f"{prefix}_histogram_overlap",
    ]
    result: dict[str, Any] = {}
    for emd_column in ("emd", "reproduced_emd", "obs_state_emd", "reproduced_obs_state_emd"):
        result[emd_column] = {
            column: {
                "pearson": float(frame[emd_column].corr(frame[column], method="pearson")),
                "spearman": float(frame[emd_column].corr(frame[column], method="spearman")),
            }
            for column in columns
        }
    return result


def _finalize_results(
    *,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    subset_path: Path,
    checkpoint_dir: Path,
    global_time: int,
    data_path: Path,
    eval_csv: Path,
    eval_timestep: int,
    edges: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    numeric_columns = [
        column
        for column in frame.columns
        if column not in selected.columns
        and column != "reproduced_emd"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    binned = (
        frame.groupby(["emd_bin_id", "emd_bin"], sort=True, observed=True)
        .agg(
            sample_count=("motion_id", "size"),
            dataset_count=("emd_bin_dataset_count", "first"),
            recorded_emd_mean=("emd", "mean"),
            recorded_emd_min=("emd", "min"),
            recorded_emd_max=("emd", "max"),
            reproduced_emd_mean=("reproduced_emd", "mean"),
            **{f"{column}_mean": (column, "mean") for column in numeric_columns},
        )
        .reset_index()
    )
    binned.to_csv(output_dir / "binned_summary.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_global_time": global_time,
        "expert_dataset": str(data_path),
        "evaluation_csv": str(eval_csv),
        "evaluation_timestep": eval_timestep,
        "seed": int(args.seed),
        "domain_randomization_disabled": True,
        "observation_noise_disabled": True,
        "selection": {
            "samples_per_nonempty_bin": int(args.samples_per_bin),
            "bin_edges": edges.tolist(),
            "num_selected_motions": int(len(frame)),
            "representative_method": "evenly spaced ranks within each recorded-EMD bin",
        },
        "interpretation": {
            "sigmoid_semantics": "P(expert | state, privileged_state, z)",
            "training_z": "expert uses mean(next 8 expert encodings); policy uses actual single-frame expert guidance z",
            "shared_guidance_z": "expert and policy both use the same actual guidance z; isolates state source",
            "roc_auc": "0.5 means no rank separation; 1.0 means complete expert-above-policy separation",
            "policy_fooled_rate_at_0.5": "fraction of policy frames classified as expert by D's actual 0.5 threshold",
        },
        "relationship": {
            prefix: _relationship_summary(frame, prefix)
            for prefix in (
                "training_z",
                "shared_guidance_z",
                "shared_expert_future_z",
                "source_future_z",
            )
        },
        "files": {
            "selected_motions": str(output_dir / "selected_motions.csv"),
            "per_motion": str(output_dir / "per_motion.csv"),
            "binned_summary": str(output_dir / "binned_summary.csv"),
            "subset_dataset": str(subset_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[EMD-DISC] wrote {summary_path}", flush=True)
    return output_dir


def analyze(args: argparse.Namespace) -> Path:
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
    eval_timestep, latest_rows = _latest_evaluation_rows(eval_csv, global_time)
    edges = _parse_bin_edges(args.bin_edges)
    selected = _select_representative_rows(latest_rows, edges, args.samples_per_bin)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder / f"emd_discriminator_relationship_{global_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    selected.to_csv(output_dir / "selected_motions.csv", index=False)
    subset_path = output_dir / "selected_motions.pkl"
    _write_subset_dataset(data_path, selected, subset_path)

    per_motion_path = output_dir / "per_motion.csv"
    if args.finalize_only:
        if not per_motion_path.is_file():
            raise FileNotFoundError(f"Missing completed rollout results: {per_motion_path}")
        frame = pd.read_csv(per_motion_path)
        if len(frame) != len(selected) or frame["motion_id"].tolist() != selected["motion_id"].tolist():
            raise ValueError("Existing per_motion.csv does not match the selected motion set")
        return _finalize_results(
            frame=frame,
            selected=selected,
            output_dir=output_dir,
            subset_path=subset_path,
            checkpoint_dir=checkpoint_dir,
            global_time=global_time,
            data_path=data_path,
            eval_csv=eval_csv,
            eval_timestep=eval_timestep,
            edges=edges,
            args=args,
        )

    _set_seed(args.seed)
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
    env._motion_lib.load_all_motions()
    env.is_evaluating = True

    results: list[dict[str, Any]] = []
    try:
        for local_motion_id, row in selected.reset_index(drop=True).iterrows():
            print(
                f"[EMD-DISC] {local_motion_id + 1}/{len(selected)} "
                f"global_motion_id={int(row['motion_id'])} recorded_emd={float(row['emd']):.6f}",
                flush=True,
            )
            metrics = _evaluate_motion(
                wrapped_env=wrapped_env,
                model=model,
                local_motion_id=local_motion_id,
                device=args.device,
            )
            record = {key: value for key, value in row.to_dict().items()}
            record.update(metrics)
            results.append(record)
            pd.DataFrame(results).to_csv(per_motion_path, index=False)
            print(
                f"[EMD-DISC] reproduced_emd={metrics['reproduced_emd']:.6f} "
                f"training_z_policy_p={metrics['training_z_policy_probability_mean']:.6f} "
                f"training_z_auc={metrics['training_z_roc_auc']:.6f} "
                f"shared_z_auc={metrics['shared_guidance_z_roc_auc']:.6f}",
                flush=True,
            )
    finally:
        wrapped_env.close()

    return _finalize_results(
        frame=pd.DataFrame(results),
        selected=selected,
        output_dir=output_dir,
        subset_path=subset_path,
        checkpoint_dir=checkpoint_dir,
        global_time=global_time,
        data_path=data_path,
        eval_csv=eval_csv,
        eval_timestep=eval_timestep,
        edges=edges,
        args=args,
    )


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
