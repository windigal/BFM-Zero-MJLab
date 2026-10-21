"""rsl-rl rollout storage with one return/advantage per reward head."""

import torch
from rsl_rl.storage import RolloutStorage


class MultiHeadRolloutStorage(RolloutStorage):
    def __init__(self, num_envs, num_transitions_per_env, obs, actions_shape, num_heads=4, device="cpu"):
        super().__init__("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.num_heads = num_heads
        for name in ("rewards", "values", "returns", "advantages"):
            setattr(self, name, torch.zeros(num_transitions_per_env, num_envs, num_heads, device=device))

    def add_transition(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer is full")
        self.observations[self.step].copy_(transition.observations)
        for name in ("actions", "rewards", "values"):
            getattr(self, name)[self.step].copy_(getattr(transition, name))
        self.dones[self.step].copy_(transition.dones.reshape(-1, 1))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.reshape(-1, 1))
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device) for p in transition.distribution_params
            )
        for buffer, value in zip(self.distribution_params, transition.distribution_params):
            buffer[self.step].copy_(value)
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, env_mask=None):
        """Optionally exclude entire validation trajectories, including every timestep."""
        indices = torch.arange(self.num_transitions_per_env * self.num_envs, device=self.device).reshape(-1, self.num_envs)
        indices = indices.flatten() if env_mask is None else indices[:, env_mask].flatten()
        if len(indices) < num_mini_batches:
            raise ValueError("Each training minibatch must contain at least one sample")
        fields = {
            "observations": self.observations.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "old_actions_log_prob": self.actions_log_prob.flatten(0, 1),
        }
        distribution = tuple(p.flatten(0, 1) for p in self.distribution_params)
        for _ in range(num_epochs):
            shuffled = indices[torch.randperm(len(indices), device=self.device)]
            for batch_ids in shuffled.tensor_split(num_mini_batches):
                yield self.Batch(
                    **{name: value[batch_ids] for name, value in fields.items()},
                    old_distribution_params=tuple(value[batch_ids] for value in distribution),
                )
