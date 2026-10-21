"""Tracking, goal/random prompting, held-out reward prompting, and cross returns."""

import json
from pathlib import Path

import torch
from tensordict import TensorDict

from .models import FbResidualActorModel
from .pretrained import EvaluationAgent, FrozenRewardModel, PretrainedController
from .runner import evaluate_tracking
from .train import build_environment, build_expert_buffer, build_parser


def load_tasks(path):
    """Task file has disjoint `train` / `eval` lists; only eval tasks are scored."""
    data = json.loads(Path(path).read_text())
    training = {task["name"] for task in data["train"]}
    evaluation = [task["name"] for task in data["eval"]]
    if training.intersection(evaluation) or len(set(evaluation)) != len(evaluation) or not evaluation:
        raise ValueError("Reward prompting requires unique held-out eval task names")
    return data["eval"]


def task_rewards(observations, tasks):
    """Explicit observation targets: exp(-mean(((x[indices]-target)/scale)^2))."""
    rewards = []
    for task in tasks:
        scale = float(task["scale"])
        if scale <= 0:
            raise ValueError("Task scale must be positive")
        if not task["indices"] or len(task["indices"]) != len(task["target"]):
            raise ValueError("Task indices/target must have equal nonzero lengths")
        values = observations[task["observation"]][:, task["indices"]]
        target = values.new_tensor(task["target"])
        rewards.append(torch.exp(-((values - target) / scale).square().mean(-1)))
    return torch.stack(rewards, dim=-1)


@torch.inference_mode()
def evaluate_latents(env, agent, latents, *, horizon=500, gamma=0.98, goal_states=None, goal_tolerance=0.2, tasks=None):
    """Group environments by fixed z; accumulate pre-reset features for cross returns.

    R[i,j] is the normalized discounted reward for z_i under policy z_j. Goal success
    requires joint-position and gravity RMS errors below goal_tolerance. Falls mean
    pelvis height < 0.3 m at any step. Counts/time-to-goal include the full horizon.
    """
    core = env.base_env
    if horizon < 1 or not 0 <= gamma <= 1 or not 1 <= len(latents) <= env.num_envs:
        raise ValueError("Invalid horizon, gamma or number of latent policies")
    groups = torch.arange(env.num_envs, device=env.device) % len(latents)
    context = latents[groups]
    reward_model = FrozenRewardModel(agent._model).to(env.device)
    previous = core.ppo_reward_model, core.ppo_latent, core.tracking_rollout_suspended
    features = torch.zeros(env.num_envs, latents.shape[-1], device=env.device)
    arrived = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    arrival = torch.zeros(env.num_envs, device=env.device)
    falls = torch.zeros_like(arrived)
    rewards = torch.zeros(env.num_envs, len(tasks), device=env.device) if tasks else None
    discount, normalizer = 1.0, 0.0
    try:
        core.tracking_rollout_suspended = True
        observation, _ = env.reset(to_numpy=False)
        core.ppo_reward_model, core.ppo_latent = reward_model, context
        for step in range(horizon):
            action = agent.act(observation, context, mean=True).clamp(-1, 1)
            observation, _, _, _, _ = env.step(action, to_numpy=False)
            features += discount * reward_model.last_embedding
            normalizer += discount
            if tasks:
                rewards += discount * task_rewards(reward_model.last_observation, tasks)
            if goal_states is not None:
                actual = reward_model.last_observation["state"]
                target = goal_states[groups]
                dofs = core.num_dof
                joints = (actual[:, :dofs] - target[:, :dofs]).square().mean(-1).sqrt()
                gravity = (actual[:, 2 * dofs : 2 * dofs + 3] - target[:, 2 * dofs : 2 * dofs + 3]).square().mean(-1).sqrt()
                reached = (joints < goal_tolerance) & (gravity < goal_tolerance)
                new = reached & ~arrived
                arrival[new] = (step + 1) * core.dt
                arrived |= reached
            # root_height is the first privileged feature, captured before any reset.
            falls |= reward_model.last_observation["privileged_state"][:, 0] < 0.3
            discount *= gamma
    finally:
        core.ppo_reward_model, core.ppo_latent, core.tracking_rollout_suspended = previous
    counts = torch.bincount(groups, minlength=len(latents)).float()
    successors = torch.zeros_like(latents).index_add_(0, groups, features / normalizer) / counts[:, None]
    cross_returns = latents @ successors.T / latents.shape[-1]
    report = {
        "cross_returns": cross_returns.cpu().tolist(),
        "mean_fb": cross_returns.diag().mean().item(),
        "fall_rate": falls.float().mean().item(),
    }
    if goal_states is not None:
        report["goal_success_rate"] = arrived.float().mean().item()
        report["mean_time_to_goal_seconds"] = arrival[arrived].mean().item() if arrived.any() else None
    if tasks:
        returns = torch.zeros(len(latents), len(tasks), device=env.device).index_add_(0, groups, rewards / normalizer) / counts[:, None]
        report["task_names"] = [task["name"] for task in tasks]
        report["prompted_task_returns"] = returns.cpu().tolist()
    return report


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--ppo-checkpoint", help="Omit to evaluate the original pretrained actor")
    parser.add_argument("--mode", choices=("tracking", "goal", "random", "prompting"), default="tracking")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--num-latents", type=int, default=8)
    parser.add_argument("--goal-tolerance", type=float, default=0.2)
    parser.add_argument("--tasks", help="JSON with disjoint train/eval task lists")
    parser.add_argument("--expert-motions", type=int, help="Limit FK preprocessing for a quick check")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    model = PretrainedController(args.init_checkpoint, args.device)
    env = build_environment(args)
    env.base_env.ppo_reward_observation_noise = args.reward_observation_noise
    try:
        actor = None
        if args.ppo_checkpoint:
            saved = torch.load(args.ppo_checkpoint, map_location=args.device, weights_only=False)
            FrozenRewardModel(model).assert_matches(saved["frozen_reward_state_dict"])
            physical, _ = env.reset(to_numpy=False)
            obs = TensorDict(physical | {"latent": model.sample_z(env.num_envs, args.device)}, batch_size=[env.num_envs])
            actor = FbResidualActorModel(
                obs, saved["config"]["obs_groups"], "actor", model.action_dim, pretrained=model, **saved["config"]["actor"]
            ).to(args.device)
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
            actor.eval()
        agent = EvaluationAgent(model, actor)
        if args.mode == "tracking":
            report = evaluate_tracking(env, model, actor, max_motions=args.eval_motions, selection_seed=args.eval_seed)
        elif args.mode == "random":
            report = evaluate_latents(env, agent, model.sample_z(args.num_latents, args.device), horizon=args.horizon)
        else:
            buffer = build_expert_buffer(args, model, args.expert_motions)
            observations = {key: value for key, value in buffer.storage["observation"].items() if key in ("state", "privileged_state")}
            if args.mode == "goal":
                indices = torch.randint(len(buffer), (args.num_latents,))
                targets = {key: value[indices].to(args.device) for key, value in observations.items()}
                context = model.goal_inference(targets)
                report = evaluate_latents(
                    env, agent, context, horizon=args.horizon, goal_states=targets["state"], goal_tolerance=args.goal_tolerance
                )
            else:
                if not args.tasks:
                    parser.error("--tasks is required for reward prompting")
                tasks = load_tasks(args.tasks)
                # Inference needs B(next_state), never actor/history inputs.
                samples = {
                    key: value[torch.arange(0, len(buffer), max(1, len(buffer) // 65536))].to(args.device)
                    for key, value in observations.items()
                }
                rewards = task_rewards(samples, tasks)
                context = torch.cat([model.reward_inference(samples, rewards[:, i : i + 1]) for i in range(len(tasks))])
                report = evaluate_latents(env, agent, context, horizon=args.horizon, tasks=tasks)
        text = json.dumps(report, indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n")
        print(text)
    finally:
        env.close()


if __name__ == "__main__":
    main()
