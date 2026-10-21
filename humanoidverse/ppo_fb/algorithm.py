"""Multi-head PPO with validated critic warmup and a fixed-rate, KL-limited actor."""

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from .models import FbResidualActorModel, MultiHeadCriticModel
from .storage import MultiHeadRolloutStorage
from .vec_env import HEADS, ROLES


class PpoFb(PPO):
    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        reward_weights=(1.0, 0.5, 0.3, 0.3),
        role_weights=None,
        consistency_coef=1.0,
        actor_learning_rate=3e-5,
        critic_learning_rate=3e-4,
        actor_epochs=2,
        warmup_iterations=50,
        warmup_max_iterations=500,
        warmup_critic_epochs=10,
        warmup_nrmse=0.2,
        warmup_bias=0.1,
        warmup_patience=5,
        warmup_motion_coverage=0.8,
        use_clipped_value_loss=False,
        **kwargs,
    ):
        if use_clipped_value_loss:
            raise ValueError("PPO-FB requires an unclipped value loss")
        super().__init__(actor, critic, storage, learning_rate=actor_learning_rate, use_clipped_value_loss=False, **kwargs)
        if self.schedule != "fixed" or self.desired_kl is None or self.desired_kl <= 0:
            raise ValueError("PPO-FB requires a fixed actor rate and a positive KL stop threshold")
        if not 0 <= warmup_iterations <= warmup_max_iterations or min(actor_epochs, warmup_critic_epochs, warmup_patience) < 1:
            raise ValueError("Invalid update or warmup iteration counts")
        if min(warmup_nrmse, warmup_bias) <= 0 or not 0 <= warmup_motion_coverage <= 1:
            raise ValueError("Invalid critic convergence thresholds")
        self.actor_parameters = [p for p in actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor_parameters, "lr": actor_learning_rate},
                {"params": critic.parameters(), "lr": critic_learning_rate},
            ]
        )
        self.reward_weights = torch.tensor(reward_weights, device=self.device)
        if self.reward_weights.shape != (storage.num_heads,) or not self.reward_weights.any():
            raise ValueError("Reward weights must match the value heads and enable at least one reward")
        self.role_weights = (
            torch.tensor(
                role_weights if role_weights is not None else [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]],
                dtype=torch.float32,
                device=self.device,
            )
            if storage.num_heads == 4
            else torch.ones(3, 1, device=self.device)
        )
        self.consistency_coef, self.actor_epochs = consistency_coef, actor_epochs
        self.warmup_iterations, self.warmup_max_iterations = warmup_iterations, warmup_max_iterations
        self.warmup_critic_epochs = warmup_critic_epochs
        self.warmup_nrmse, self.warmup_bias = warmup_nrmse, warmup_bias
        self.warmup_patience, self.warmup_motion_coverage = warmup_patience, warmup_motion_coverage
        self.iteration, self.validation_streak = 0, 0
        self.warming_up = warmup_iterations > 0
        self.validation_envs = torch.zeros(storage.num_envs, dtype=torch.bool, device=self.device)
        self._set_actor_trainable()

    def _set_validation_envs(self, roles):
        self.validation_envs.zero_()
        for role in roles.unique():
            ids = (roles == role).nonzero(as_tuple=False).flatten()
            if len(ids) < 2:
                raise ValueError("Critic warmup needs at least two environments per enabled role")
            self.validation_envs[ids[::8]] = True

    def _set_actor_trainable(self):
        for parameter in self.actor_parameters:
            parameter.requires_grad_(not self.warming_up)
        self.optimizer.param_groups[0]["lr"] = 0.0 if self.warming_up else self.learning_rate

    def process_env_step(self, obs, rewards, dones, extras):
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.transition.rewards = rewards.clone() + self.gamma * self.transition.values * extras["time_outs"].to(self.device)[:, None]
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, obs):
        self.last_observations = obs
        st = self.storage
        next_values = self.critic(obs).detach()
        advantage = torch.zeros_like(next_values)
        # Long lambda=1 rollouts evaluate the fixed policy during warmup. Fresh
        # rollouts refresh the bootstrap targets after every critic update.
        lam = 1.0 if self.warming_up else self.lam
        for step in reversed(range(st.num_transitions_per_env)):
            continuing = 1 - st.dones[step].float()
            delta = st.rewards[step] + continuing * self.gamma * next_values - st.values[step]
            advantage = delta + continuing * self.gamma * lam * advantage
            st.returns[step] = advantage + st.values[step]
            next_values = st.values[step]
        advantages = st.returns - st.values
        st.advantages.copy_((advantages - advantages.mean((0, 1))) / (advantages.std((0, 1)) + 1e-8))

    @torch.no_grad()
    def _validate_critic(self, *, advance_streak=True):
        # Validate BEFORE fitting, on separate environments and only the first
        # half of the rollout, where the bootstrapped tail is furthest away.
        st = self.storage
        horizon = max(1, st.num_transitions_per_env // 2)
        obs = st.observations[:horizon, self.validation_envs].flatten(0, 1)
        targets = st.returns[:horizon, self.validation_envs].flatten(0, 1)
        values = torch.cat([self.critic(part) for part in obs.split(1024)])
        metrics, passed = {}, True
        for role in obs["role"].argmax(-1).unique().tolist():
            mask = obs["role"][:, role].bool()
            for head in range(st.num_heads):
                if self.reward_weights[head] == 0 or self.role_weights[role, head] == 0:
                    continue
                target, error = targets[mask, head], values[mask, head] - targets[mask, head]
                scale = target.square().mean().sqrt().clamp_min(1.0)
                nrmse, bias = error.square().mean().sqrt() / scale, error.mean().abs() / scale
                name = f"validation/{ROLES[role]}/{HEADS[head] if st.num_heads == 4 else 'total'}"
                metrics[f"{name}/nrmse"], metrics[f"{name}/bias"] = nrmse.item(), bias.item()
                metrics[f"{name}/target_mean"] = target.mean().item()
                metrics[f"{name}/prediction_mean"] = values[mask, head].mean().item()
                metrics[f"{name}/explained_variance"] = (1 - error.var(unbiased=False) / target.var(unbiased=False).clamp_min(1e-8)).item()
                passed &= bool(nrmse <= self.warmup_nrmse and bias <= self.warmup_bias)
        coverage = self.expert_motion_coverage()
        coverage_ok = coverage >= self.warmup_motion_coverage or not (st.observations[0]["role"][:, 0] > 0).any()
        passed &= coverage_ok and self.iteration + 1 >= self.warmup_iterations
        if not passed:
            self.validation_streak = 0
        elif advance_streak:
            self.validation_streak += 1
        metrics["warmup/motion_coverage"] = coverage
        metrics["warmup/validation_streak"] = self.validation_streak
        return metrics

    def _update_critic(self, warmup):
        totals = torch.zeros(self.storage.num_heads, device=self.device)
        epochs = self.warmup_critic_epochs if warmup else self.num_learning_epochs
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, epochs, ~self.validation_envs if warmup else None):
            losses = (self.critic(batch.observations) - batch.returns).square().mean(0)
            active = self.reward_weights != 0
            loss = self.value_loss_coef * losses[active].mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals += losses.detach()
            updates += 1
        return totals / updates, updates

    def _update_actor(self):
        totals = torch.zeros(7, device=self.device)
        updates, stopped = 0, False
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.actor_epochs):
            obs = batch.observations
            self.actor.distribution.update(self.actor(obs))
            kl = self.actor.get_kl_divergence(batch.old_distribution_params, self.actor.output_distribution_params).mean()
            if kl > self.desired_kl:
                stopped = True
                break
            log_prob = self.actor.get_output_log_prob(batch.actions)
            advantage = (batch.advantages * (obs["role"] @ self.role_weights) * self.reward_weights).sum(-1)
            ratio = (log_prob - batch.old_actions_log_prob.flatten()).exp()
            surrogate = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)).mean()
            mask = 1 - obs["role"][:, 0]
            distance = (self.actor.output_mean - self.actor.reference_mean(obs)).square().mean(-1)
            consistency = (distance * mask).sum() / mask.sum().clamp_min(1)
            entropy = self.actor.output_entropy.mean()
            loss = surrogate - self.entropy_coef * entropy + self.consistency_coef * consistency
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
            self.optimizer.step()
            totals += torch.stack(
                (
                    surrogate.detach(),
                    entropy.detach(),
                    kl.detach(),
                    consistency.detach(),
                    ((ratio - 1).abs() > self.clip_param).float().mean().detach(),
                    distance.mean().detach(),
                    loss.detach(),
                )
            )
            updates += 1
        return totals / max(updates, 1), updates, stopped

    @torch.no_grad()
    def _rollout_kl(self):
        total, count = 0.0, 0
        observations = self.storage.observations.flatten(0, 1)
        distribution = tuple(p.flatten(0, 1) for p in self.storage.distribution_params)
        for start in range(0, len(observations), 1024):
            obs = observations[start : start + 1024]
            self.actor.distribution.update(self.actor(obs))
            old = tuple(p[start : start + len(obs)] for p in distribution)
            kl = self.actor.get_kl_divergence(old, self.actor.output_distribution_params)
            total += kl.sum().item()
            count += kl.numel()
        return total / count

    def update(self):
        warmup = self.warming_up
        metrics = self._validate_critic() if warmup else {}
        value_losses, critic_updates = self._update_critic(warmup)
        if warmup and self.validation_streak >= self.warmup_patience:
            metrics.update(self._validate_critic(advance_streak=False))
        actor_metrics, actor_updates, stopped = (torch.zeros(7), 0, False) if warmup else self._update_actor()
        metrics.update(
            zip(
                ("surrogate", "entropy", "minibatch_kl", "consistency", "clip_fraction", "reference_distance", "actor_loss"),
                actor_metrics.tolist(),
            )
        )
        metrics["kl"] = 0.0 if warmup else self._rollout_kl()
        metrics["actor_updates"], metrics["critic_updates"], metrics["kl_stopped"] = actor_updates, critic_updates, float(stopped)
        active = self.reward_weights != 0
        metrics["value"] = value_losses[active].mean().item()
        for i, value in enumerate(value_losses.tolist()):
            head = HEADS[i] if self.storage.num_heads == 4 else "total"
            metrics[f"value/{head}"] = value
            metrics[f"prediction/{head}"] = self.storage.values[:, :, i].mean().item()
        metrics["actor_learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["critic_learning_rate"] = self.optimizer.param_groups[1]["lr"]
        metrics["warmup/active"] = float(warmup)
        if str(self.device).startswith("cuda"):
            metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        self.storage.clear()
        self.iteration += 1
        if warmup and self.validation_streak >= self.warmup_patience:
            self.warming_up = False
            self._set_actor_trainable()
        return metrics

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        if cfg["multi_gpu"] is not None:
            raise ValueError("PPO-FB currently supports a single training process")
        actor = FbResidualActorModel(obs, cfg["obs_groups"], "actor", env.num_actions, pretrained=env.pretrained, **cfg["actor"]).to(device)
        num_heads = 1 if env.single_head else 4
        critic = MultiHeadCriticModel(obs, cfg["obs_groups"], "critic", num_heads, **cfg["critic"]).to(device)
        storage = MultiHeadRolloutStorage(env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], num_heads, device)
        options = {key: value for key, value in cfg["algorithm"].items() if key not in ("class_name", "rnd_cfg")}
        alg = PpoFb(
            actor, critic, storage, device=device, reward_weights=[1.0] if env.single_head else env.reward_weights.tolist(), **options
        )
        if alg.warming_up:
            alg._set_validation_envs(obs["role"].argmax(-1))
        alg.expert_motion_coverage = env.expert_motion_coverage
        return alg

    def save(self):
        return super().save() | {
            "ppo_fb_iteration": self.iteration,
            "actor_learning_rate": self.learning_rate,
            "warming_up": self.warming_up,
            "validation_streak": self.validation_streak,
        }

    @torch.inference_mode()
    def load(self, loaded_dict, load_cfg=None, strict=True):
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.iteration = loaded_dict["ppo_fb_iteration"]
            self.learning_rate = loaded_dict["actor_learning_rate"]
            self.warming_up = loaded_dict["warming_up"]
            self.validation_streak = loaded_dict["validation_streak"]
            self._set_actor_trainable()
        return load_iteration
