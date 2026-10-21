"""rsl-rl models preserving the pretrained actor's input and output conventions."""

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from torch import nn

from humanoidverse.agents.nn_models import ResidualBlock


class FbResidualActorModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, pretrained, init_std=0.05):
        # MLPModel's distribution/state API is reused; no unused generic MLP is allocated.
        nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = False
        self.obs_normalizer = copy.deepcopy(pretrained._obs_normalizer).requires_grad_(False).eval()
        self.mlp = copy.deepcopy(pretrained._actor).requires_grad_(True)
        self.reference_actor = copy.deepcopy(pretrained._actor).requires_grad_(False).eval()
        self.distribution = GaussianDistribution(output_dim, init_std=init_std, std_type="log", std_range=(0.02, 0.3))

    def train(self, mode=True):
        super().train(mode)
        self.obs_normalizer.eval()
        self.reference_actor.eval()
        return self

    def update_normalization(self, obs):
        pass

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = self.mlp(self.obs_normalizer(dict(obs.items())), obs["latent"], 1.0).mean
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def reference_mean(self, obs):
        return self.reference_actor(self.obs_normalizer(dict(obs.items())), obs["latent"], 1.0).mean


class MultiHeadCriticModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, hidden_dim=1024, hidden_layers=3):
        nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = True
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        self.distribution = None
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Mish(),
            *(ResidualBlock(hidden_dim) for _ in range(hidden_layers)),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
