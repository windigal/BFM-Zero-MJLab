# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

import gymnasium
import torch
from torch.amp import autocast

from ..fb_cpr.model import FBcprModel, FBcprModelArchiConfig, FBcprModelConfig
from ..nn_models import ForwardArchiConfig, RewardNormalizerConfig


class FBcprAuxModelArchiConfig(FBcprModelArchiConfig):
    aux_critic: ForwardArchiConfig = ForwardArchiConfig()
    tracking_critic: ForwardArchiConfig = ForwardArchiConfig()


class FBcprAuxModelConfig(FBcprModelConfig):
    name: tp.Literal["FBcprAuxModel"] = "FBcprAuxModel"
    archi: FBcprAuxModelArchiConfig = FBcprAuxModelArchiConfig()
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig()
    use_tracking_critic: bool = False

    @property
    def object_class(self):
        return FBcprAuxModel


class FBcprAuxModel(FBcprModel):
    config_class = FBcprAuxModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: FBcprAuxModelConfig):
        # NOTE for future: if we inherit models, we need to make sure that the cfg we pass in (which is wrong)
        #      can still be used to build the underlying models
        super().__init__(obs_space, action_dim, cfg)
        self.cfg = cfg
        self._aux_critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()
        if cfg.use_tracking_critic:
            tracking_spaces = dict(obs_space.spaces)
            tracking_spaces["reference_state"] = copy.deepcopy(obs_space.spaces["state"])
            tracking_spaces["reference_privileged_state"] = copy.deepcopy(obs_space.spaces["privileged_state"])
            self._tracking_obs_space = gymnasium.spaces.Dict(tracking_spaces)
            self._tracking_critic = cfg.archi.tracking_critic.build(
                self._tracking_obs_space,
                cfg.archi.z_dim,
                action_dim,
                output_dim=1,
            )
        else:
            self._tracking_obs_space = None
            self._tracking_critic = None

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_aux_critic = copy.deepcopy(self._aux_critic)
        self._target_tracking_critic = (
            copy.deepcopy(self._tracking_critic) if self._tracking_critic is not None else None
        )

    @torch.no_grad()
    def aux_critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._aux_critic(self._normalize(obs), z, action)
