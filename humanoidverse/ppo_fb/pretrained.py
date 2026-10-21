"""Selectively load the parts of FB-CPR needed for on-policy control."""

import json

import torch
import torch.nn.functional as F
from torch import nn

from humanoidverse.agents.base_model import BaseModel
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.fb.model import FBModel, FBModelArchiConfig, FBModelConfig
from humanoidverse.agents.frozen_fb import load_checkpoint_tensors, resolve_fb_checkpoint
from humanoidverse.agents.nn_models import ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig
from humanoidverse.agents.normalizers import ObsNormalizerConfig


class PretrainedController(FBModel):
    """FB inference interface without allocating F or any of the off-policy critics."""

    def __init__(self, checkpoint, device="cpu"):
        root = resolve_fb_checkpoint(checkpoint) / "model"
        config = json.loads((root / "config.json").read_text())
        kwargs = json.loads((root / "init_kwargs.json").read_text())
        arch = config["archi"]
        cfg = FBModelConfig(
            device=torch.device(device).type,
            archi=FBModelArchiConfig(
                z_dim=arch["z_dim"],
                norm_z=arch["norm_z"],
                b=BackwardArchiConfig(**arch["b"]),
                actor=ActorArchiConfig(**arch["actor"]),
            ),
            obs_normalizer=ObsNormalizerConfig(**config["obs_normalizer"]),
            seq_length=config["seq_length"],
            actor_std=config["actor_std"],
            amp=config["amp"],
        )
        obs_space = json_to_space(kwargs["obs_space"])
        BaseModel.__init__(self, obs_space, kwargs["action_dim"], cfg)
        self.device = torch.device(device).type
        self.amp_dtype = torch.bfloat16
        self._backward_map = cfg.archi.b.build(obs_space, arch["z_dim"])
        self._obs_normalizer = cfg.obs_normalizer.build(obs_space)
        self._actor = cfg.archi.actor.build(obs_space, arch["z_dim"], self.action_dim)
        self._discriminator = DiscriminatorArchiConfig(**arch["discriminator"]).build(obs_space, arch["z_dim"])
        state = load_checkpoint_tensors(checkpoint, ("_backward_map.", "_obs_normalizer.", "_actor.", "_discriminator."))
        self.load_state_dict(state, strict=True)
        self.to(device).requires_grad_(False).eval()


class FrozenRewardModel(nn.Module):
    """Rewards from the true next state, evaluated by the core before reset."""

    def __init__(self, pretrained: PretrainedController, inverse_second_moment=None):
        super().__init__()
        self.normalizer = pretrained._obs_normalizer
        self.backward = pretrained._backward_map
        self.discriminator = pretrained._discriminator
        self.register_buffer("inverse_second_moment", inverse_second_moment)
        self.last_embedding = None
        self.last_observation = None
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def assert_matches(self, state):
        mismatched = [key for key, value in self.state_dict().items() if not torch.equal(value, state[key].to(value.device))]
        if mismatched:
            raise ValueError(f"Checkpoint frozen rewards differ at {mismatched[:3]}; use its --init-checkpoint and inverse Sigma")

    @torch.no_grad()
    def forward(self, observation, latent):
        normalized = self.normalizer(observation)
        embedding = self.backward(normalized).float()
        self.last_embedding = embedding
        self.last_observation = observation
        if self.inverse_second_moment is None:
            fb = F.cosine_similarity(embedding, latent, dim=-1)
        else:
            fb = ((embedding @ self.inverse_second_moment) * latent).sum(-1) / latent.shape[-1]
        disc = self.discriminator.compute_logits(normalized, latent).squeeze(-1).clamp(-5, 5) / 5
        return torch.stack((fb, disc), dim=-1)


class EvaluationAgent:
    """Existing tracking evaluation's model/agent protocol."""

    def __init__(self, pretrained, actor=None):
        self._model = pretrained
        self.actor = actor

    @torch.no_grad()
    def act(self, observation, latent, mean=True):
        if self.actor is None:
            return self._model.act(observation, latent, mean=mean)
        from tensordict import TensorDict

        obs = TensorDict({**observation, "latent": latent}, batch_size=[latent.shape[0]])
        return self.actor(obs, stochastic_output=not mean)
