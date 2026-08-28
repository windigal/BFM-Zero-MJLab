"""UFO training entrypoint.

UFO provides FB and TeCH unsupervised RL presets for humanoid control.
Defaults are kept in this file; command-line arguments can override them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _ensure_compile_cache(cache_root: str | Path | None = None) -> None:
    cache_dir = os.environ.get("UFO_CACHE_DIR") or os.environ.get("BFMZERO_MJLAB_CACHE_DIR")
    root = Path(cache_dir or cache_root or Path.cwd() / "cache").expanduser().resolve()
    for key, subdir in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "WARP_CACHE_PATH": "warp",
    }.items():
        configured_path = os.environ.get(key)
        path = Path(configured_path).expanduser().resolve() if configured_path else root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


_ensure_compile_cache()

DEFAULT_AGENT = "fb"
DEFAULT_NUM_ENVS = 1024
DEFAULT_NUM_ENV_STEPS = 192000000
DEFAULT_CHECKPOINT_EVERY_STEPS = 3200000
DEFAULT_DATA_PATH = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
DEFAULT_WORK_DIR = "runs/bfm_zero_seed"
DEFAULT_BUFFER_SIZE = 5120000
DEFAULT_FB_UPDATE_Z_EVERY_STEP = 100
DEFAULT_TECH_UPDATE_Z_EVERY_STEP = 10
DEFAULT_UPDATE_Z_EVERY_STEP = DEFAULT_FB_UPDATE_Z_EVERY_STEP
DEFAULT_WANDB_ENTITY = "windigal"
DEFAULT_WANDB_PROJECT = "bfm-zero"
DEFAULT_WANDB_GROUP = "mjlab"
DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"

AGENT_ALIASES = {
    "fb": "fb",
    "tech": "tech",
    "tldr": "tech",
}

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.training.workspace import PACEConfig, TrainConfig
from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path


def _resolve_training_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def canonical_agent_name(agent: str) -> str:
    try:
        return AGENT_ALIASES[agent]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_ALIASES))
        raise ValueError(f"Unsupported agent preset: {agent}. Supported presets: {supported}") from exc


def _default_update_z_every_step(agent: str) -> int:
    canonical = canonical_agent_name(agent)
    return DEFAULT_TECH_UPDATE_Z_EVERY_STEP if canonical == "tech" else DEFAULT_FB_UPDATE_Z_EVERY_STEP


def build_ufo_mjlab_config(
    *,
    device: str,
    work_dir: str,
    num_envs: int,
    num_env_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
    checkpoint_every_steps: int = 9600000,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
    num_agent_updates: int | None = None,
    robot_config: str | Path | None = None,
    emd_group_low_threshold: float = 0.5,
    emd_group_high_threshold: float = 2.0,
    emd_high_group_fraction: float | None = None,
    pace_base_checkpoint: str | Path | None = None,
    tracking_q: bool = False,
    init_checkpoint: str | Path | None = None,
    tracking_actor_grad_ratio: float = 0.10,
    disable_discriminator: bool = False,
) -> TrainConfig:
    agent = canonical_agent_name(agent)
    robot_training = load_robot_training_spec(robot_config or DEFAULT_ROBOT_CONFIG)
    try:
        raw_robot_config = OmegaConf.to_container(OmegaConf.load(robot_training.config_path), resolve=True)
        metadata = raw_robot_config.get("metadata") if isinstance(raw_robot_config, dict) else None
        if isinstance(metadata, dict) and metadata.get("review_status") == "draft":
            print(
                "WARNING: Robot config is auto-generated draft. Review semantics, default pose, PD gains, "
                "actuator parameters, contact bodies, and reward/termination-related fields before formal training.",
                flush=True,
            )
    except Exception as exc:
        print(f"WARNING: Could not inspect robot config metadata for draft status: {exc}", flush=True)
    evaluations = []
    pace_cfg = PACEConfig(base_checkpoint=str(Path(pace_base_checkpoint).expanduser().resolve())) if pace_base_checkpoint else None
    run_eval_and_prioritization = pace_cfg is not None or (not smoke and not disable_eval_prioritization)
    distributed_sync = distributed_world_size > 1
    if run_eval_and_prioritization:
        evaluations = [
            HumanoidVerseMjlabTrackingEvaluationConfig(
                name="HumanoidVerseMjlabTrackingEvaluationConfig",
                generate_videos=False,
                videos_dir="videos",
                video_name_prefix="unknown_agent",
                name_in_logs="humanoidverse_tracking_eval",
                env=None,
                num_envs=num_envs,
                n_episodes_per_motion=1,
            )
        ]
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    resolved_update_z_every_step = (
        _default_update_z_every_step(agent) if update_z_every_step is None else int(update_z_every_step)
    )
    selected = build_agent_preset(
        agent=agent,
        device=agent_device,
        compile=not distributed_sync,
        update_z_every_step=resolved_update_z_every_step,
        lr_scale=lr_scale,
        clip_grad_norm=clip_grad_norm,
        cartwheel_aux_safe=cartwheel_aux_safe,
        enable_tracking_q=tracking_q,
        freeze_backward=bool(tracking_q and init_checkpoint is not None),
        tracking_actor_grad_ratio=tracking_actor_grad_ratio,
        disable_discriminator=disable_discriminator,
        wandb_project=DEFAULT_WANDB_PROJECT,
        wandb_group=DEFAULT_WANDB_GROUP,
    )
    agent_cfg = selected["agent_cfg"]
    wandb_group = selected["wandb_group"]
    wandb_project = selected["wandb_project"]
    train_runtime = dict(selected["train_runtime"])
    if num_agent_updates is not None:
        if num_agent_updates <= 0:
            raise ValueError("num_agent_updates must be positive")
        train_runtime["num_agent_updates"] = int(num_agent_updates)
    if pace_cfg is not None:
        # Stage II starts from a trained policy. Fill one complete PACE horizon
        # with policy rollouts, rather than the legacy random-action warmup.
        train_runtime["num_seed_steps"] = int(num_envs * pace_cfg.rollout_horizon)
        # FB-CPR's acquisition objective contains unnormalized Q values. PACE
        # therefore uses a deliberately smaller actor LR in addition to the
        # runtime warmup/ramp; critic, discriminator, and F learning rates stay
        # unchanged so they can adapt to the Stage-II rollout distribution.
        pace_agent_train = agent_cfg.train.model_copy(
            update={"lr_actor": float(agent_cfg.train.lr_actor) * pace_cfg.actor_lr_scale}
        )
        agent_cfg = agent_cfg.model_copy(update={"train": pace_agent_train})
    hydra_overrides = [
        f"robot={robot_training.hydra_robot}",
        f"robot.control.action_scale={robot_training.action_scale}",
        f"robot.control.action_clip_value={robot_training.action_clip_value}",
        f"robot.control.normalize_action_to={robot_training.normalize_action_to}",
        *robot_training.hydra_overrides,
    ]
    if cartwheel_aux_safe:
        hydra_overrides.extend(
            [
                "rewards.reward_scales.penalty_undesired_contact=0.0",
                "rewards.reward_scales.penalty_feet_ori=0.0",
                "rewards.reward_scales.feet_heading_alignment=0.0",
                "rewards.reward_scales.penalty_slippage=0.0",
                "rewards.reward_scales.penalty_ankle_roll=0.0",
                "rewards.reward_scales.penalty_action_rate=-0.1",
            ]
        )

    return TrainConfig(
        name="TrainConfig",
        agent=agent_cfg,
        motions="",
        motions_root="",
        env=HumanoidVerseMjlabConfig(
            name="humanoidverse_mjlab",
            device=device,
            lafan_tail_path=data_path or DEFAULT_DATA_PATH,
            data_mix_weights=data_mix_weights,
            mjcf_path=robot_training.robot.xml_path,
            robot_config_path=str(robot_training.config_path),
            robot_training=robot_training.to_env_dict(),
            max_episode_length_s=None,
            disable_obs_noise=disable_obs_noise,
            disable_domain_randomization=disable_dr,
            relative_config_path="exp/bfm_zero/bfm_zero",
            include_last_action=True,
            hydra_overrides=hydra_overrides,
            context_length=None,
            include_history_actor=True,
            include_history_noaction=False,
            root_height_obs=True,
            auto_reset=False,
            seed=seed,
        ),
        work_dir=work_dir,
        seed=seed,
        online_parallel_envs=num_envs,
        log_every_updates=train_runtime["log_every_updates"],
        num_env_steps=num_env_steps,
        update_agent_every=train_runtime["update_agent_every"],
        num_seed_steps=train_runtime["num_seed_steps"],
        num_agent_updates=train_runtime["num_agent_updates"],
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_buffer=train_runtime["checkpoint_buffer"],
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=float(emd_group_low_threshold),
        prioritization_max_val=float(emd_group_high_threshold),
        prioritization_scale=2.0,
        prioritization_mode="exp",
        prioritization_high_group_ratio=0.80 if pace_cfg is not None else 0.05,
        prioritization_high_group_fraction=(None if pace_cfg is not None else emd_high_group_fraction),
        pace=pace_cfg,
        init_checkpoint=(str(Path(init_checkpoint).expanduser().resolve()) if init_checkpoint else None),
        use_trajectory_buffer=train_runtime["use_trajectory_buffer"],
        buffer_size=int(buffer_size),
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
        wandb_gname=wandb_group,
        wandb_pname=wandb_project,
        wandb_run_name=wandb_run_name or f"ufo_{agent}",
        load_expert_data_from_motion_lib=True,
        buffer_device="cuda" if device.startswith("cuda") else "cpu",
        expert_buffer_device="cpu",
        disable_tqdm=True,
        evaluations=evaluations,
        eval_every_steps=train_runtime["eval_every_steps"],
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        rank0_only_writes=True,
        checkpoint_rank_buffers=True,
        distributed_sync=distributed_sync,
        distributed_global_steps=True,
        distributed_average_metrics=True,
        tags={"backend": "mjlab", "agent": agent, "distributed_rank": distributed_rank, "distributed_world_size": distributed_world_size},
    )


def _select_device_and_rank(seed: int) -> tuple[str, int, int, int]:
    # torchrunx sets the distributed rank variables in each worker even when
    # the parent shell has no CUDA_VISIBLE_DEVICES override.  Inspect them
    # first; otherwise every worker falls through to the single-GPU shortcut
    # below and tries to allocate on cuda:0.
    del seed
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise RuntimeError("LOCAL_RANK, RANK, and WORLD_SIZE must be integers") from exc
    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")
    if world_size > 1:
        missing = [name for name in ("LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Incomplete distributed environment (missing "
                + ", ".join(missing)
                + "). Unset stale rank variables or launch through torchrunx/torchrun."
            )
        if not 0 <= rank < world_size:
            raise RuntimeError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE, got rank={rank}, world_size={world_size}")
        if local_rank < 0:
            raise RuntimeError(f"LOCAL_RANK must be non-negative, got {local_rank}")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Distributed NCCL training requires CUDA, but torch.cuda.is_available() is False")
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {device_count} visible CUDA device(s)"
            )
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        return f"cuda:{local_rank}", local_rank, rank, world_size

    # Ignore stale single-process rank values and always select logical cuda:0.
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
            return "cuda:0", 0, 0, 1
    except Exception:
        pass
    return "cpu", 0, 0, 1


def _init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timedelta(hours=2),
        }
        try:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)


def _new_timestamped_run_dir(work_dir: Path, timestamp: str | None = None) -> Path:
    """Create one unique timestamp-named run directory below ``work_dir``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        name = timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}"
        candidate = work_dir / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            pass
        suffix += 1


def _distributed_timestamped_run_dir(
    work_dir: Path,
    *,
    local_rank: int,
    rank: int,
    world_size: int,
) -> Path:
    """Create one shared timestamped directory for an external torchrun job."""
    if world_size <= 1:
        return _new_timestamped_run_dir(work_dir)

    import torch
    import torch.distributed as dist

    payload: list[str | None] = [None]
    if rank == 0:
        payload[0] = _new_timestamped_run_dir(work_dir).name
    dist.broadcast_object_list(
        payload,
        src=0,
        device=torch.device(f"cuda:{local_rank}"),
    )
    if payload[0] is None:
        raise RuntimeError("Rank 0 did not broadcast a timestamped run directory")
    run_dir = work_dir / payload[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_train(args: argparse.Namespace, log_dir: Path, log_dir_is_final: bool = False) -> None:
    device, _local_rank, rank, world_size = _select_device_and_rank(args.seed)
    _init_distributed(_local_rank, world_size)
    if not log_dir_is_final:
        log_dir = _distributed_timestamped_run_dir(
            log_dir,
            local_rank=_local_rank,
            rank=rank,
            world_size=world_size,
        )
    seed = args.seed + rank
    cfg = build_ufo_mjlab_config(
        device=device,
        work_dir=str(log_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=seed,
        use_wandb=bool(args.use_wandb and rank == 0),
        wandb_run_name=args.wandb_run_name or log_dir.name,
        checkpoint_every_steps=args.checkpoint_every_steps,
        distributed_rank=rank,
        distributed_world_size=world_size,
        disable_eval_prioritization=bool(args.disable_eval_prioritization),
        smoke=bool(args.smoke),
        agent=args.agent,
        data_path=args.data_path,
        data_mix_weights=args.data_mix_weights,
        update_z_every_step=args.update_z_every_step,
        buffer_size=args.buffer_size,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        lr_scale=args.lr_scale,
        clip_grad_norm=args.clip_grad_norm,
        cartwheel_aux_safe=bool(args.cartwheel_aux_safe),
        num_agent_updates=args.num_agent_updates,
        robot_config=args.robot_config,
        emd_group_low_threshold=args.emd_group_low_threshold,
        emd_group_high_threshold=args.emd_group_high_threshold,
        emd_high_group_fraction=args.emd_high_group_fraction,
        pace_base_checkpoint=args.pace_base_checkpoint,
        tracking_q=bool(args.tracking_q),
        init_checkpoint=args.init_checkpoint,
        tracking_actor_grad_ratio=args.tracking_actor_grad_ratio,
        disable_discriminator=bool(args.disable_discriminator),
    )
    print(
        "[INFO] UFO train: "
        f"agent={args.agent}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"robot_config={cfg.env.robot_config_path}, mjcf_path={cfg.env.mjcf_path}, "
        f"data_path={cfg.env.lafan_tail_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"num_agent_updates={cfg.num_agent_updates}, update_agent_every_local={cfg.update_agent_every}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"pace_base_checkpoint={args.pace_base_checkpoint}, "
        f"tracking_q={args.tracking_q}, tracking_actor_grad_ratio={args.tracking_actor_grad_ratio}, "
        f"disable_discriminator={args.disable_discriminator}, "
        f"init_checkpoint={args.init_checkpoint}, "
        "emd_sampling=global_all_trajectories, "
        f"emd_weight_clamp=[{args.emd_group_low_threshold}, {args.emd_group_high_threshold}], "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"compile={cfg.agent.compile}",
        flush=True,
    )
    try:
        workspace = cfg.build()
        workspace.train()
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache()
    if args.gpu_ids in (None, "single"):
        run_train(args, work_dir)
        return

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        import torch

        num_gpus = torch.cuda.device_count()
        selected_gpus = None
    else:
        requested = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        if existing_visible:
            visible = [x.strip() for x in existing_visible.split(",") if x.strip()]
            selected_gpus = [visible[i] for i in requested]
        else:
            selected_gpus = [str(i) for i in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        run_train(args, work_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    log_dir = _new_timestamped_run_dir(work_dir)
    os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        # Let torchrunx create the NCCL process group. torchrunx populates the
        # standard rank variables for every backend; the explicit NCCL backend
        # also initializes the collective group before run_train starts.
        backend="nccl",
        # The rank-0-only full-dataset tracking evaluation currently takes a
        # little over ten minutes for 5k motions / 512 envs.  torchrunx creates
        # the process group before run_train, so its default 600-second timeout
        # wins over the two-hour timeout in _init_distributed and aborts the
        # other ranks while they legitimately wait at the post-eval barrier.
        worker_timeout=2 * 60 * 60,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
        + (
            "MUJOCO*",
            "UFO_CACHE_DIR",
            "BFMZERO_MJLAB_CACHE_DIR",
            "UV_CACHE_DIR",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
            "TEMP",
            "TMP",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "WARP_CACHE_PATH",
        ),
    ).run(run_train, args, log_dir, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFO.")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=["fb", "tech", "tldr"],
        help="Training agent preset: fb or tech. tldr is a deprecated alias for tech.",
    )
    parser.add_argument("--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help=(
            "Parent directory for training runs. Each launch creates a new "
            "YYYYMMDD_HHMMSS child directory."
        ),
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "Robot YAML used for training metadata. Defaults to configs/robots/g1_29dof.yaml. "
            "If omitted and --data-manifest declares robot_config, the manifest robot config is used."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help="One or more motion data pickle files. Multiple files require --data-mix-weights to fix source ratios.",
    )
    parser.add_argument(
        "--data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help="Source-level sampling weights for multiple --data-path entries, e.g. 0.95 0.05.",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="YAML manifest describing weighted motion data sources. Cannot be combined with --data-path.",
    )
    parser.add_argument(
        "--rebuild-motion-cache",
        action="store_true",
        help="Rebuild manifest-generated motion pkl caches instead of reusing existing cache files.",
    )
    parser.add_argument(
        "--update-z-every-step",
        type=int,
        default=None,
        help="Override latent update interval. Defaults to 100 for FB and 10 for TeCH.",
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay capacity per rank/GPU.")
    parser.add_argument(
        "--num-agent-updates",
        type=int,
        default=None,
        help=(
            "Override optimizer updates per update trigger. For fair env-scaling ablations, use 32 with "
            "2048 envs/GPU and 64 with 4096 envs/GPU to match the 1024 envs/GPU update density."
        ),
    )
    parser.add_argument("--disable-dr", action="store_true", help="Disable domain randomization for training.")
    parser.add_argument("--disable-obs-noise", action="store_true", help="Disable observation noise for training.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale FB learning rates. TeCH preset ignores this value.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Enable FB actor/FB gradient clipping when > 0.")
    parser.add_argument(
        "--emd-group-low-threshold",
        type=float,
        default=0.5,
        help="Lower clamp used by global exponential EMD weighting.",
    )
    parser.add_argument(
        "--emd-group-high-threshold",
        type=float,
        default=2.0,
        help="Upper clamp used by global exponential EMD weighting.",
    )
    parser.add_argument(
        "--emd-high-group-fraction",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option. Single-dataset training now samples all trajectories from one "
            "global EMD-weighted distribution and ignores high-EMD group fractions."
        ),
    )
    parser.add_argument(
        "--pace-base-checkpoint",
        type=Path,
        default=None,
        help=(
            "Enable PACE Stage-II training and initialize model weights from this Stage-I run or checkpoint directory. "
            "Optimizers, replay, and global step start fresh."
        ),
    )
    parser.add_argument(
        "--tracking-q",
        action="store_true",
        help=(
            "Enable the Extreme-RGMT/BeyondMimic reference-tracking Q. Aligned expert environments "
            "start from the exact first reference state and roll out the complete motion; the actor "
            "observation interface is unchanged."
        ),
    )
    parser.add_argument(
        "--tracking-actor-grad-ratio",
        type=float,
        default=0.10,
        help=(
            "Target norm ratio of the weighted Tracking-Q actor gradient to the base actor gradient. "
            "For example, 0.30 targets 30%%."
        ),
    )
    parser.add_argument(
        "--disable-discriminator",
        action="store_true",
        help=(
            "Skip discriminator/critic training and drop the discriminator term from the actor loss. "
            "Modules stay constructed so checkpoints remain compatible. Requires --tracking-q."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize model weights only from a run or checkpoint directory. Optimizers, replay, and "
            "training counters start fresh. With --tracking-q, B/target-B/observation normalization are frozen."
        ),
    )
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help=(
            "Use a cartwheel-safe FB auxiliary reward set: remove undesired-contact, foot-orientation, "
            "feet-heading, slippage, and ankle-roll penalties, and reduce action-rate penalties."
        ),
    )
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    raw_agent = args.agent
    args.agent = canonical_agent_name(args.agent)
    if args.pace_base_checkpoint is not None and args.init_checkpoint is not None:
        parser.error("--pace-base-checkpoint and --init-checkpoint are mutually exclusive")
    if args.tracking_q and args.agent != "fb":
        parser.error("--tracking-q is supported only with --agent fb")
    if args.tracking_q and args.init_checkpoint is None and args.pace_base_checkpoint is None:
        parser.error("--tracking-q requires --init-checkpoint (or --pace-base-checkpoint for later PACE use)")
    if not np.isfinite(args.tracking_actor_grad_ratio) or args.tracking_actor_grad_ratio < 0.0:
        parser.error("--tracking-actor-grad-ratio must be finite and non-negative")
    if args.disable_discriminator and not args.tracking_q:
        parser.error("--disable-discriminator requires --tracking-q (Stage-II tracking mode)")
    if args.emd_high_group_fraction is not None and (
        not np.isfinite(args.emd_high_group_fraction) or not 0 < args.emd_high_group_fraction < 1
    ):
        parser.error("--emd-high-group-fraction must be finite and strictly between 0 and 1")
    if raw_agent == "tldr":
        print("WARNING: agent=tldr is deprecated; use agent=tech instead.", file=sys.stderr, flush=True)
    if args.update_z_every_step is None:
        args.update_z_every_step = _default_update_z_every_step(args.agent)
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.num_env_steps = min(args.num_env_steps, 2048)
        args.use_wandb = False
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        manifest_data = prepare_motion_manifest(args.data_manifest, rebuild_cache=bool(args.rebuild_motion_cache))
        args.data_path = manifest_data.train_data_paths
        args.data_mix_weights = manifest_data.train_data_weights
        manifest_robot_config = manifest_data.robot_config_path
    elif args.data_path is not None:
        data_path_count = len(args.data_path)
        if args.data_mix_weights is not None:
            if len(args.data_mix_weights) != data_path_count:
                raise ValueError("--data-mix-weights length must match --data-path length")
            if any(w < 0 for w in args.data_mix_weights) or sum(args.data_mix_weights) <= 0:
                raise ValueError("--data-mix-weights must be non-negative and sum to a positive value")
            weight_sum = float(sum(args.data_mix_weights))
            args.data_mix_weights = [float(w) / weight_sum for w in args.data_mix_weights]
        elif data_path_count > 1:
            args.data_mix_weights = [1.0 / data_path_count] * data_path_count
        if data_path_count == 1:
            args.data_path = args.data_path[0]
            args.data_mix_weights = None

    args.robot_config = _resolve_training_robot_config(args.robot_config, manifest_robot_config)

    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.num_agent_updates is not None and args.num_agent_updates <= 0:
        raise ValueError("--num-agent-updates must be positive")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if not np.isfinite(args.emd_group_low_threshold) or not np.isfinite(args.emd_group_high_threshold):
        raise ValueError("EMD group thresholds must be finite")
    if not 0 <= args.emd_group_low_threshold < args.emd_group_high_threshold:
        raise ValueError("EMD group thresholds must satisfy 0 <= low < high")
    if args.cartwheel_aux_safe and args.agent != "fb":
        raise ValueError("--cartwheel-aux-safe is only supported with --agent fb")
    if args.pace_base_checkpoint is not None:
        if args.agent != "fb":
            raise ValueError("--pace-base-checkpoint is only supported with --agent fb")
        if args.disable_eval_prioritization:
            raise ValueError("PACE cannot be combined with --disable-eval-prioritization")
        if isinstance(args.data_path, list) and len(args.data_path) != 1:
            raise ValueError("PACE requires exactly one complete motion dataset")
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
