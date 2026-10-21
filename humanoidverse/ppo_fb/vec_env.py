"""Aligned expert, goal and random latent rollouts for rsl-rl."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from humanoidverse.agents.frozen_fb import encode_expert_latents

from .pretrained import FrozenRewardModel

HEADS = ("track", "fb", "disc", "aux")
ROLES = ("expert", "goal", "random")


@dataclass
class PpoFbEnvConfig:
    role_fractions: tuple
    latent_interval: int


class ExpertReference:
    """CPU expert observations and precomputed per-motion rolling z windows."""

    def __init__(self, buffer, pretrained):
        self.observations = {key: buffer.storage["observation"][key].cpu() for key in ("state", "privileged_state")}
        self.starts = buffer.start_idx.flatten().cpu()
        self.lengths = buffer.lengths.flatten().cpu()
        if buffer.motion_ids != list(range(len(self.lengths))):
            raise ValueError("PPO expert data must contain all global motion ids in order")
        self.latents = torch.zeros(len(buffer), pretrained.cfg.archi.z_dim)
        for start, length in zip(self.starts.tolist(), self.lengths.tolist()):
            observation = {key: value[start + 1 : start + length] for key, value in self.observations.items()}
            self.latents[start : start + length - 1] = encode_expert_latents(
                pretrained,
                observation,
                seq_length=pretrained.cfg.seq_length,
            ).cpu()

    def at(self, motion_ids, steps, device):
        ids, steps = motion_ids.cpu(), steps.cpu()
        if torch.any(steps < 0) or torch.any(steps >= self.lengths[ids] - 1):
            raise IndexError("Expert rollout step has no next reference frame")
        indices = self.starts[ids] + steps
        reference = torch.cat([self.observations[key][indices + 1] for key in ("state", "privileged_state")], dim=-1)
        return self.latents[indices].to(device), reference.to(device)

    def sample_goals(self, count, pretrained, device):
        ids = torch.randint(len(self.lengths), (count,))
        steps = (torch.rand(count) * (self.lengths[ids] - 1)).long() + 1
        indices = self.starts[ids] + steps
        observations = {key: value[indices].to(device) for key, value in self.observations.items()}
        return pretrained.project_z(pretrained.backward_map(observations).float())


class PpoFbVecEnv(VecEnv):
    def __init__(
        self,
        env,
        pretrained,
        experts,
        *,
        role_fractions=(0.5, 0.25, 0.25),
        latent_interval=100,
        truncate_latent_switch=True,
        reward_weights=(1.0, 0.5, 0.3, 0.3),
        single_head=False,
        inverse_second_moment=None,
        reward_observation_noise=False,
    ):
        if len(role_fractions) != 3 or min(role_fractions) < 0 or abs(sum(role_fractions) - 1) > 1e-6:
            raise ValueError("Three nonnegative role fractions must sum to one")
        if latent_interval < 1:
            raise ValueError("latent_interval must be positive")
        self.env, self.core, self.pretrained, self.experts = env, env.base_env, pretrained, experts
        self.num_envs, self.num_actions, self.device = env.num_envs, env.base_env.num_dof, env.device
        self.max_episode_length = round(float(self.core.config.max_episode_length_s) / self.core.dt)
        # A serializable env config also works with rsl-rl's W&B writer.
        self.cfg = PpoFbEnvConfig(tuple(role_fractions), latent_interval)
        expert_end = int(self.num_envs * role_fractions[0])
        goal_end = expert_end + int(self.num_envs * role_fractions[1])
        self.roles = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)
        self.roles[:expert_end] = 0
        self.roles[expert_end:goal_end] = 1
        self.latent_interval, self.truncate_latent_switch = latent_interval, truncate_latent_switch
        self.reward_weights = torch.tensor(reward_weights, device=self.device)
        if self.reward_weights.shape != (4,):
            raise ValueError("Four reward weights are required")
        self.single_head = single_head
        self.latent = torch.zeros(self.num_envs, pretrained.cfg.archi.z_dim, device=self.device)
        self.latent_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference = torch.zeros(
            self.num_envs,
            sum(value.shape[-1] for value in experts.observations.values()),
            device=self.device,
        )
        self.core.configure_tracking_rollouts(
            self.roles == 0,
            expert_motion_lengths=experts.lengths,
            seq_length=pretrained.cfg.seq_length,
        )
        self.core.ppo_reward_model = FrozenRewardModel(pretrained, inverse_second_moment).to(self.device)
        self.core.ppo_reward_observation_noise = reward_observation_noise
        self.core.ppo_latent = self.latent
        self.seen_expert_motions = torch.zeros(len(experts.lengths), dtype=torch.bool, device=self.device)
        self.reset()

    def expert_motion_coverage(self):
        return self.seen_expert_motions.float().mean().item()

    @property
    def episode_length_buf(self):
        return self.core.episode_length_buf

    def _refresh_expert(self):
        ids = (self.roles == 0).nonzero(as_tuple=False).flatten()
        if ids.numel():
            motion_ids = self.core._motion_lib.global_motion_ids(self.core.motion_ids[ids])
            latent, reference = self.experts.at(motion_ids, self.episode_length_buf[ids], self.device)
            self.latent[ids], self.reference[ids] = latent, reference

    def _resample(self, mask):
        goal_ids = (mask & (self.roles == 1)).nonzero(as_tuple=False).flatten()
        random_ids = (mask & (self.roles == 2)).nonzero(as_tuple=False).flatten()
        if goal_ids.numel():
            self.latent[goal_ids] = self.experts.sample_goals(len(goal_ids), self.pretrained, self.device)
        if random_ids.numel():
            self.latent[random_ids] = self.pretrained.sample_z(len(random_ids), device=self.device)
        self.latent_age[mask] = 0

    def _observations(self, physical):
        return TensorDict(
            {key: physical[key] for key in ("state", "privileged_state", "last_action", "history_actor")}
            | {"latent": self.latent.clone(), "reference": self.reference.clone(), "role": F.one_hot(self.roles, 3).float()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    @torch.inference_mode()
    def reset(self):
        physical, _ = self.env.reset(to_numpy=False)
        self.latent_age.zero_()
        self._refresh_expert()
        self._resample(self.roles != 0)
        self._obs = self._observations(physical)
        return self._obs

    def get_observations(self):
        # Querying observations twice must not advance history or mutate rollout inputs.
        return self._obs

    @torch.inference_mode()
    def step(self, actions):
        expert_ids = self.core.motion_ids[self.roles == 0]
        self.seen_expert_motions[self.core._motion_lib.global_motion_ids(expert_ids)] = True
        physical, _, terminated, truncated, info = self.env.step(actions.clamp(-1, 1), to_numpy=False)
        reset = terminated | truncated
        aux = info["aux_rewards"]
        rewards = torch.stack(
            (
                info["tracking_rewards"]["total"],
                info["representation_rewards"][:, 0],
                info["representation_rewards"][:, 1],
                -0.03 * aux["penalty_action_rate"] - 10 * aux["limits_dof_pos"],
            ),
            dim=-1,
        )
        self.latent_age += 1
        switch = (self.roles != 0) & (self.latent_age >= self.latent_interval) & ~reset
        self._resample((self.roles != 0) & (reset | switch))
        self._refresh_expert()
        self._obs = self._observations(physical)
        log = {f"Reward/{name}": rewards[:, i].mean() for i, name in enumerate(HEADS)}
        for role, name in enumerate(ROLES):
            mask = self.roles == role
            if mask.any():
                for head, head_name in enumerate(HEADS):
                    log[f"Reward/{name}/{head_name}"] = rewards[mask, head].mean()
        synthetic_timeout = switch if self.truncate_latent_switch else torch.zeros_like(switch)
        extras = {"time_outs": (truncated & ~terminated) | synthetic_timeout, "log": log}
        if self.single_head:
            rewards = (rewards * self.reward_weights).sum(-1, keepdim=True)
        return self._obs, rewards, reset | synthetic_timeout, extras

    def close(self):
        self.core.ppo_reward_model = None
        self.core.ppo_latent = None
        self.env.close()
