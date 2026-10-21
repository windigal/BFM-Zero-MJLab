"""Train PPO-FB: ./run_ppo_fb.sh --init-checkpoint runs/pretrained/.../checkpoint."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

# Configure Warp/compile caches before importing simulator/runtime libraries.
from humanoidverse.train import _ensure_compile_cache, build_ufo_mjlab_config

_ensure_compile_cache()

import numpy as np
import torch

from humanoidverse.agents.envs.expert_motion_loader import load_expert_trajectories_from_motion_lib

from .pretrained import PretrainedController
from .runner import PpoFbRunner
from .vec_env import ExpertReference, PpoFbVecEnv

DEFAULT_CHECKPOINT = "runs/pretrained/20260810_002844/checkpoint"
DEFAULT_DATA = "humanoidverse/data/seed_train_full10s_5k.pkl"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume")
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--work-dir", default=f"runs/{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-env-steps", type=int, default=50_000_000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--actor-epochs", type=int, default=2)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--warmup-max-iterations", type=int, default=500)
    parser.add_argument("--warmup-rollout-steps", type=int, default=128)
    parser.add_argument("--warmup-critic-epochs", type=int, default=10)
    parser.add_argument("--warmup-nrmse", type=float, default=0.2)
    parser.add_argument("--warmup-bias", type=float, default=0.1)
    parser.add_argument("--warmup-patience", type=int, default=5)
    parser.add_argument("--warmup-motion-coverage", type=float, default=0.8)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--init-std", type=float, default=0.05)
    parser.add_argument("--consistency-coef", type=float, default=1.0)
    parser.add_argument("--reward-weights", type=float, nargs=4, default=(1, 0.5, 0.3, 0.3))
    parser.add_argument("--role-fractions", type=float, nargs=3, default=(0.5, 0.25, 0.25))
    parser.add_argument("--latent-interval", type=int, default=100)
    parser.add_argument("--no-truncate-latent-switch", action="store_true")
    parser.add_argument("--single-head", action="store_true")
    parser.add_argument("--critic-hidden-dim", type=int, default=1024)
    parser.add_argument("--critic-hidden-layers", type=int, default=3)
    parser.add_argument("--inverse-second-moment", help="Frozen inverse Sigma tensor from inspect_pretrained.py")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000, help="PPO iterations between evaluations; 0 disables evaluation")
    parser.add_argument("--eval-motions", type=int, help="Limit evaluation motions; omitted evaluates the full dataset")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--motion-pool-refresh-steps", type=int, default=1024, help="Per-environment control steps between pool refreshes")
    parser.add_argument("--motion-pool-size", type=int, default=1024, help="Reference motions resident on GPU, independent of environment count")
    parser.add_argument("--disable-dr", action="store_true")
    parser.add_argument("--disable-obs-noise", action="store_true")
    parser.add_argument("--reward-observation-noise", action="store_true", help="Ablation: also add actor observation noise to B/D rewards")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Log to both TensorBoard and W&B")
    parser.add_argument("--wandb-project", default="ppo-fb")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Compare pretrained and random actions without PPO updates")
    return parser


def build_env_config(args):
    config = build_ufo_mjlab_config(
        device=args.device,
        work_dir=args.work_dir,
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=args.seed,
        use_wandb=False,
        wandb_run_name=None,
        data_path=args.data_path,
        smoke=True,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
    ).env
    config.hydra_overrides.append(f"+robot.motion.override_num_motions_to_load={args.motion_pool_size}")
    return config


def build_environment(args):
    env, _ = build_env_config(args).build(num_envs=args.num_envs)
    return env


def build_expert_buffer(args, pretrained, max_motions=None):
    """Forward-kinematics expert observations without creating a simulator (CPU supported)."""
    from humanoidverse.agents.envs.humanoidverse_mjlab import _compose_humanoidverse_config, _default_joint_pos
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

    cfg = build_env_config(args)
    config, _ = _compose_humanoidverse_config(
        num_envs=args.num_envs,
        relative_config_path=cfg.relative_config_path,
        hydra_overrides=cfg.hydra_overrides,
        lafan_tail_path=cfg.lafan_tail_path,
        data_mix_weights=cfg.data_mix_weights,
        disable_obs_noise=cfg.disable_obs_noise,
        disable_domain_randomization=cfg.disable_domain_randomization,
        max_episode_length_s=cfg.max_episode_length_s,
        root_height_obs=cfg.root_height_obs,
        robot_training=cfg.robot_training,
    )
    dt = int(config.simulator.config.sim.control_decimation) / float(config.simulator.config.sim.fps)
    config.robot.motion.step_dt = dt
    reader = SimpleNamespace(
        config=config,
        num_envs=args.num_envs,
        device=args.device,
        dt=dt,
        default_dof_pos=_default_joint_pos(config).to(args.device)[None],
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]], device=args.device),
        _motion_lib=MotionLibRobot(config.robot.motion, num_envs=args.num_envs, device=args.device),
    )
    return load_expert_trajectories_from_motion_lib(
        reader,
        SimpleNamespace(model=pretrained.cfg),
        device="cpu",
        max_motions=max_motions,
    )


def build_runner_config(args):
    return {
        "run": vars(args).copy(),
        "num_steps_per_env": args.rollout_steps,
        "warmup_rollout_steps": args.warmup_rollout_steps,
        "joint_rollout_steps": args.rollout_steps,
        "max_env_steps": args.num_env_steps,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "eval_motions": args.eval_motions,
        "eval_seed": args.eval_seed,
        "motion_pool_refresh_steps": args.motion_pool_refresh_steps,
        "motion_pool_size": args.motion_pool_size,
        "seed": args.seed,
        "logger": "wandb" if args.wandb else "tensorboard",
        "wandb_project": args.wandb_project,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "data_path": args.data_path,
        "obs_groups": {
            "actor": ["state", "last_action", "history_actor", "latent"],
            "critic": ["state", "last_action", "history_actor", "privileged_state", "latent", "reference", "role"],
        },
        "actor": {"init_std": args.init_std},
        "critic": {"hidden_dim": args.critic_hidden_dim, "hidden_layers": args.critic_hidden_layers},
        "algorithm": {
            "class_name": "humanoidverse.ppo_fb.algorithm:PpoFb",
            "rnd_cfg": None,
            "num_learning_epochs": args.epochs,
            "num_mini_batches": args.minibatches,
            "clip_param": 0.2,
            "gamma": 0.98,
            "lam": 0.95,
            "entropy_coef": 0.0,
            "max_grad_norm": 1.0,
            "desired_kl": args.target_kl,
            "schedule": "fixed",
            "use_clipped_value_loss": False,
            "actor_epochs": args.actor_epochs,
            "actor_learning_rate": args.actor_lr,
            "critic_learning_rate": args.critic_lr,
            "warmup_iterations": args.warmup_iterations,
            "warmup_max_iterations": args.warmup_max_iterations,
            "warmup_critic_epochs": args.warmup_critic_epochs,
            "warmup_nrmse": args.warmup_nrmse,
            "warmup_bias": args.warmup_bias,
            "warmup_patience": args.warmup_patience,
            "warmup_motion_coverage": args.warmup_motion_coverage,
            "consistency_coef": args.consistency_coef,
        },
    }


@torch.inference_mode()
def smoke(env, steps):
    results = {}
    for random_actions in (False, True):
        observation = env.reset()
        total = torch.zeros(4, device=env.device)
        start = time.perf_counter()
        for _ in range(steps):
            actions = (
                torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
                if random_actions
                else env.pretrained.act(
                    dict(observation.items()),
                    observation["latent"],
                    mean=True,
                )
            )
            observation, rewards, _, _ = env.step(actions)
            if not torch.isfinite(rewards).all() or not all(torch.isfinite(value).all() for value in observation.values()):
                raise RuntimeError("Non-finite observation/reward in smoke rollout")
            total += rewards.mean(0)
        results["random" if random_actions else "pretrained"] = {
            "mean_rewards": (total / steps).tolist(),
            "env_steps_per_second": steps * env.num_envs / (time.perf_counter() - start),
        }
    return results


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        min(
            args.num_envs,
            args.num_env_steps,
            args.rollout_steps,
            args.warmup_rollout_steps,
            args.epochs,
            args.minibatches,
            args.save_interval,
            args.motion_pool_refresh_steps,
            args.motion_pool_size,
        )
        < 1
    ):
        raise ValueError("Training sizes and save interval must be positive")
    if args.num_envs * args.rollout_steps < args.minibatches:
        raise ValueError("Each PPO minibatch must contain at least one sample")
    if args.smoke_steps and args.single_head:
        raise ValueError("Smoke reward diagnostics require the four separate heads")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; check GPU visibility and the NVIDIA driver")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[INFO] PPO-FB run directory: {args.work_dir}", flush=True)
    pretrained = PretrainedController(args.init_checkpoint, args.device)
    physical_env = build_environment(args)
    try:
        buffer = load_expert_trajectories_from_motion_lib(
            physical_env.base_env,
            SimpleNamespace(model=pretrained.cfg),
            device="cpu",
        )
        experts = ExpertReference(buffer, pretrained)
        del buffer
        if int(args.num_envs * args.role_fractions[0]) and args.motion_pool_refresh_steps < experts.lengths.max().item() - 1:
            raise ValueError("Motion pool refresh must allow at least one complete expert motion")
        inverse = (
            torch.load(args.inverse_second_moment, weights_only=True, map_location=args.device) if args.inverse_second_moment else None
        )
        env = PpoFbVecEnv(
            physical_env,
            pretrained,
            experts,
            role_fractions=args.role_fractions,
            latent_interval=args.latent_interval,
            truncate_latent_switch=not args.no_truncate_latent_switch,
            reward_weights=args.reward_weights,
            single_head=args.single_head,
            inverse_second_moment=inverse,
            reward_observation_noise=args.reward_observation_noise,
        )
        if args.smoke_steps:
            print(json.dumps(smoke(env, args.smoke_steps), indent=2))
            return
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config = build_runner_config(args)
        (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        runner = PpoFbRunner(env, config, str(work_dir), args.device)
        if args.resume:
            runner.load(args.resume)
        shortest_rollout = min(args.rollout_steps, args.warmup_rollout_steps)
        remaining = math.ceil((args.num_env_steps - runner.logger.tot_timesteps) / (args.num_envs * shortest_rollout))
        if remaining > 0:
            runner.learn(remaining)
    finally:
        physical_env.base_env.ppo_reward_model = None
        physical_env.close()


if __name__ == "__main__":
    main()
