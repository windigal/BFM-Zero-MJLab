"""Run the training-style tracking evaluation and export per-motion EMD.

This intentionally reuses ``HumanoidVerseMjlabTrackingEvaluation`` so the
rollout, target construction, and EMD definition stay aligned with the metric
reported during training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluationConfig,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every motion in a dataset and export its tracking EMD."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--disable-dr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable domain randomization. Default false matches training-time eval.",
    )
    parser.add_argument(
        "--disable-obs-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable observation noise. Default false matches training-time eval.",
    )
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable rollout progress bars.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _default_output_dir(model_folder: Path, data_path: Path, global_time: int) -> Path:
    return model_folder / f"eval_{data_path.stem}_latest_{global_time}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(args: argparse.Namespace) -> Path:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    data_path = args.data_path.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing motion dataset: {data_path}")
    if args.num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {args.num_envs}")

    run_config = _read_json(model_folder / "config.json")
    train_status = _read_json(checkpoint_dir / "train_status.json")
    global_time = int(train_status.get("global_time", train_status.get("time", 0)))
    seed = int(run_config.get("seed", 0) if args.seed is None else args.seed)
    _set_seed(seed)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(model_folder, data_path, global_time)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "tracking_metrics.csv"
    emd_csv = output_dir / "emd_per_motion.csv"
    summary_json = output_dir / "summary.json"
    existing = [path for path in (metrics_csv, emd_csv, summary_json) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to append to an existing evaluation: "
            + ", ".join(str(path) for path in existing)
        )

    print(
        "[EMD-EVAL] "
        f"checkpoint={checkpoint_dir} global_time={global_time} data={data_path} "
        f"num_envs={args.num_envs} device={args.device} seed={seed} "
        f"disable_dr={args.disable_dr} disable_obs_noise={args.disable_obs_noise}",
        flush=True,
    )

    env_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=args.device,
        headless=True,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        max_episode_length_s=args.max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=args.num_envs)
    try:
        model = load_model_from_checkpoint_dir(
            checkpoint_dir, device=checkpoint_load_device(args.device)
        )
        model.to(args.device)
        model.eval()

        evaluator = HumanoidVerseMjlabTrackingEvaluationConfig(
            name="HumanoidVerseMjlabTrackingEvaluationConfig",
            env=None,
            num_envs=args.num_envs,
            n_episodes_per_motion=1,
            include_results_from_all_envs=False,
            disable_tqdm=args.disable_progress,
        ).build()
        metrics, aggregate = evaluator.run(
            timestep=global_time,
            agent_or_model=model,
            logger=CSVLogger(metrics_csv),
            env=wrapped_env,
        )
        expected_count = int(wrapped_env._env._motion_lib._num_unique_motions)
    finally:
        wrapped_env.close()

    frame = pd.read_csv(metrics_csv).sort_values("motion_id").reset_index(drop=True)
    if len(frame) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} metric rows, but found {len(frame)} in {metrics_csv}"
        )
    if frame["motion_id"].nunique() != expected_count:
        raise RuntimeError("motion_id values are not unique in the evaluation output")
    if frame["emd"].isna().any():
        raise RuntimeError("At least one motion has a missing EMD value")

    emd_frame = frame[["motion_id", "motion_name", "motion_file", "emd"]].copy()
    emd_frame.to_csv(emd_csv, index=False)
    emd = emd_frame["emd"].to_numpy(dtype=np.float64)
    worst = emd_frame.nlargest(20, "emd").to_dict(orient="records")
    best = emd_frame.nsmallest(20, "emd").to_dict(orient="records")
    summary = {
        "model_folder": str(model_folder),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_global_time": global_time,
        "checkpoint_optimizer_steps": int(train_status.get("optimizer_steps", 0)),
        "data_path": str(data_path),
        "device": args.device,
        "num_envs": int(args.num_envs),
        "seed": seed,
        "disable_domain_randomization": bool(args.disable_dr),
        "disable_observation_noise": bool(args.disable_obs_noise),
        "num_motions": int(emd.size),
        "metric": "training_style_joint_position_emd",
        "emd": {
            "mean": float(emd.mean()),
            "std": float(emd.std()),
            "min": float(emd.min()),
            "p25": float(np.percentile(emd, 25)),
            "median": float(np.median(emd)),
            "p75": float(np.percentile(emd, 75)),
            "p90": float(np.percentile(emd, 90)),
            "p95": float(np.percentile(emd, 95)),
            "p99": float(np.percentile(emd, 99)),
            "max": float(emd.max()),
        },
        "aggregate_metrics": {key: float(value) for key, value in aggregate.items()},
        "best_20": best,
        "worst_20": worst,
        "files": {
            "all_tracking_metrics": str(metrics_csv),
            "per_motion_emd": str(emd_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if len(metrics) != expected_count:
        raise RuntimeError(
            f"Evaluator returned {len(metrics)} motions, expected {expected_count}"
        )
    print(json.dumps({"num_motions": summary["num_motions"], "emd": summary["emd"]}), flush=True)
    print(f"[EMD-EVAL] wrote {emd_csv}", flush=True)
    print(f"[EMD-EVAL] wrote {summary_json}", flush=True)
    return output_dir


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
