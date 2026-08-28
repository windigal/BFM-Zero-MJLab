# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Dict

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from ...distributed import average_gradients, is_distributed
from .model import FBcprAuxModelConfig


class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5
    lr_tracking_critic: float = 3e-4
    tracking_critic_pessimism_penalty: float = 0.5
    tracking_actor_grad_ratio: float = 0.10
    tracking_actor_warmup_updates: int = 1000
    tracking_actor_ramp_updates: int = 1000
    tracking_grad_monitor_every_updates: int = 32
    tracking_grad_ema_beta: float = 0.95
    tracking_grad_scale_min: float = 1.0e-4
    tracking_grad_scale_max: float = 1.0e4
    tracking_grad_scale_rate_limit: float = 2.0


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return FBcprAuxAgent


class FBcprAuxAgent(FBcprAgent):
    config_class = FBcprAuxAgentConfig

    def __init__(self, obs_space, action_dim: int, cfg: FBcprAuxAgentConfig):
        super().__init__(obs_space, action_dim, cfg)
        self._tracking_update_count = 0
        self._tracking_grad_base_ema = 0.0
        self._tracking_grad_q_ema = 0.0
        self._tracking_grad_ema_initialized = False
        self._tracking_grad_scale = 1.0
        self._tracking_grad_cosine = 0.0
        self._tracking_component_grad_metrics: dict[str, float] = {}

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        if self._model._tracking_critic is not None:
            self._tracking_critic_map_paramlist = tuple(self._model._tracking_critic.parameters())
            self._tracking_target_critic_map_paramlist = tuple(self._model._target_tracking_critic.parameters())
            self.tracking_critic_optimizer = torch.optim.Adam(
                self._model._tracking_critic.parameters(),
                lr=self.cfg.train.lr_tracking_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay,
            )
        else:
            self._tracking_critic_map_paramlist = ()
            self._tracking_target_critic_map_paramlist = ()
            self.tracking_critic_optimizer = None

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        if self.tracking_critic_optimizer is not None:
            optimizers["tracking_critic_optimizer"] = self.tracking_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)
            if self.tracking_critic_optimizer is not None:
                self.update_tracking_critic = torch.compile(self.update_tracking_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        if self._pace_enabled:
            train_batch = replay_buffer["train"].sample_valid(self.cfg.train.batch_size)
        else:
            train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        if not self.backward_frozen:
            self._model._obs_normalizer(train_obs)
            self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        rollout_z = train_batch["z"].to(self.device)
        next_rollout_z = train_batch.get("next", {}).get("z")
        if next_rollout_z is not None:
            next_rollout_z = next_rollout_z.to(self.device)
        train_z = rollout_z

        # train the discriminator (skipped entirely in no-discriminator mode)
        if self.cfg.train.disable_discriminator:
            metrics: Dict[str, torch.Tensor] = {}
        else:
            grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
            metrics = self.update_discriminator(
                expert_obs=expert_obs,
                expert_z=expert_z,
                train_obs=train_obs,
                train_z=train_z,
                grad_penalty=grad_penalty,
            )

        if not self._pace_enabled:
            z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
            self.z_buffer.add(z)

            if self.cfg.train.relabel_ratio is not None:
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
                train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        if not self.cfg.train.disable_discriminator:
            metrics.update(
                self.update_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    z=train_z,
                )
            )
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)

        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        tracking_actor_obs = None
        tracking_next_obs = None
        tracking_actor_z = None
        if self._model._tracking_critic is not None:
            tracking_mask = train_batch["expert_z_rollout"].to(self.device).reshape(-1).bool()
            if not bool(tracking_mask.any()):
                raise RuntimeError("Tracking-Q replay batch contains no aligned expert-z transitions")
            if next_rollout_z is None:
                raise RuntimeError("Tracking-Q replay is missing next z")
            tracking_actor_obs = self._build_tracking_observation(
                obs=train_obs,
                mask=tracking_mask,
                expert_buffer=replay_buffer["expert_slicer"],
                motion_ids=train_batch["reference_motion_id"].reshape(-1)[
                    tracking_mask.to(train_batch["reference_motion_id"].device)
                ],
                reference_steps=train_batch["reference_step"].reshape(-1)[
                    tracking_mask.to(train_batch["reference_step"].device)
                ],
            )
            tracking_next_obs = self._build_tracking_observation(
                obs=train_next_obs,
                mask=tracking_mask,
                expert_buffer=replay_buffer["expert_slicer"],
                motion_ids=train_batch["next"]["reference_motion_id"].reshape(-1)[
                    tracking_mask.to(train_batch["next"]["reference_motion_id"].device)
                ],
                reference_steps=train_batch["next"]["reference_step"].reshape(-1)[
                    tracking_mask.to(train_batch["next"]["reference_step"].device)
                ],
            )
            tracking_actor_z = rollout_z[tracking_mask]
            tracking_reward = train_batch["tracking_rewards"]["total"].to(self.device)[tracking_mask]
            metrics.update(
                self.update_tracking_critic(
                    obs=tracking_actor_obs,
                    action=train_action[tracking_mask],
                    discount=discount[tracking_mask],
                    tracking_reward=tracking_reward,
                    next_obs=tracking_next_obs,
                    z=tracking_actor_z,
                    next_z=next_rollout_z[tracking_mask],
                )
            )
            for reward_name, reward_value in train_batch["tracking_rewards"].items():
                metrics[f"tracking_reward/{reward_name}"] = reward_value.to(self.device)[tracking_mask].mean()
            metrics["tracking_q/expert_batch_fraction"] = tracking_mask.float().mean()
        if self._pace_enabled:
            pace_role = train_batch["pace_role"].to(self.device).reshape(-1).bool()
            metrics.update(
                self.update_actor_pace(
                    obs=train_obs,
                    z=train_z,
                    acquisition_mask=pace_role,
                    clip_grad_norm=clip_grad_norm,
                    tracking_obs=tracking_actor_obs,
                    tracking_z=tracking_actor_z,
                )
            )
            metrics.update(
                {key: torch.tensor(value, device=self.device, dtype=torch.float32) for key, value in self._pace_runtime_metrics.items()}
            )
        else:
            metrics.update(
                self.update_actor(
                    obs=train_obs,
                    action=train_action,
                    z=train_z,
                    clip_grad_norm=clip_grad_norm,
                    tracking_obs=tracking_actor_obs,
                    tracking_z=tracking_actor_z,
                )
            )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            if not self.backward_frozen:
                _soft_update_params(
                    self._backward_map_paramlist,
                    self._target_backward_map_paramlist,
                    self.cfg.train.fb_target_tau,
                )
            if not self.cfg.train.disable_discriminator:
                _soft_update_params(
                    self._critic_map_paramlist,
                    self._target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            if self._tracking_critic_map_paramlist:
                _soft_update_params(
                    self._tracking_critic_map_paramlist,
                    self._tracking_target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        return metrics

    def _build_tracking_observation(
        self,
        *,
        obs: dict[str, torch.Tensor],
        mask: torch.Tensor,
        expert_buffer,
        motion_ids: torch.Tensor,
        reference_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reference = expert_buffer.observations_at(motion_ids, reference_steps)
        reference = tree_map(lambda value: value.to(self.device), reference)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            reference = self._model._obs_normalizer(reference)
        tracking_obs = tree_map(lambda value: value[mask], obs)
        tracking_obs = dict(tracking_obs)
        tracking_obs["reference_state"] = reference["state"]
        tracking_obs["reference_privileged_state"] = reference["privileged_state"]
        return tracking_obs

    def tracking_checkpoint_state(self) -> dict[str, tp.Any]:
        return {
            "update_count": int(self._tracking_update_count),
            "grad_base_ema": float(self._tracking_grad_base_ema),
            "grad_q_ema": float(self._tracking_grad_q_ema),
            "grad_ema_initialized": bool(self._tracking_grad_ema_initialized),
            "grad_scale": float(self._tracking_grad_scale),
            "grad_cosine": float(self._tracking_grad_cosine),
            "component_grad_metrics": dict(self._tracking_component_grad_metrics),
        }

    def load_tracking_checkpoint_state(self, state: dict[str, tp.Any] | None) -> None:
        if not state:
            return
        self._tracking_update_count = int(state.get("update_count", 0))
        self._tracking_grad_base_ema = float(state.get("grad_base_ema", 0.0))
        self._tracking_grad_q_ema = float(state.get("grad_q_ema", 0.0))
        self._tracking_grad_ema_initialized = bool(state.get("grad_ema_initialized", False))
        self._tracking_grad_scale = float(state.get("grad_scale", 1.0))
        self._tracking_grad_cosine = float(state.get("grad_cosine", 0.0))
        self._tracking_component_grad_metrics = {
            str(key): float(value) for key, value in state.get("component_grad_metrics", {}).items()
        }

    def _tracking_actor_coefficient(self) -> float:
        completed = self._tracking_update_count - int(self.cfg.train.tracking_actor_warmup_updates)
        if completed <= 0:
            return 0.0
        ramp = min(completed / max(float(self.cfg.train.tracking_actor_ramp_updates), 1.0), 1.0)
        return float(self.cfg.train.tracking_actor_grad_ratio) * ramp * self._tracking_grad_scale

    def _update_tracking_gradient_scale(
        self,
        base_loss: torch.Tensor,
        tracking_loss: torch.Tensor,
    ) -> None:
        base_norm, tracking_norm, cosine = self._gradient_statistics(
            base_loss,
            tracking_loss,
            self._model._actor.parameters(),
        )
        base_value = float(base_norm.item())
        tracking_value = float(tracking_norm.item())
        self._tracking_grad_cosine = float(cosine.item())
        if base_value <= 1.0e-12 or tracking_value <= 1.0e-12:
            return
        beta = float(self.cfg.train.tracking_grad_ema_beta)
        initialized = self._tracking_grad_ema_initialized
        if initialized:
            self._tracking_grad_base_ema = beta * self._tracking_grad_base_ema + (1.0 - beta) * base_value
            self._tracking_grad_q_ema = beta * self._tracking_grad_q_ema + (1.0 - beta) * tracking_value
        else:
            self._tracking_grad_base_ema = base_value
            self._tracking_grad_q_ema = tracking_value
            self._tracking_grad_ema_initialized = True
        target = self._tracking_grad_base_ema / max(self._tracking_grad_q_ema, 1.0e-12)
        target = min(max(target, self.cfg.train.tracking_grad_scale_min), self.cfg.train.tracking_grad_scale_max)
        if initialized:
            limit = float(self.cfg.train.tracking_grad_scale_rate_limit)
            target = min(max(target, self._tracking_grad_scale / limit), self._tracking_grad_scale * limit)
        self._tracking_grad_scale = float(target)

    def _update_tracking_gradient_diagnostics(
        self,
        *,
        base_loss: torch.Tensor,
        component_losses: dict[str, torch.Tensor],
        tracking_loss: torch.Tensor,
    ) -> None:
        """Measure each actor objective against Tracking-Q on a low-frequency batch."""
        parameters = tuple(self._model._actor.parameters())
        tracking_gradients = torch.autograd.grad(
            tracking_loss, parameters, retain_graph=True, allow_unused=True
        )
        base_gradients = torch.autograd.grad(
            base_loss, parameters, retain_graph=True, allow_unused=True
        )
        component_gradients = {
            name: torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            for name, loss in component_losses.items()
        }

        def compare(first, second) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            first_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            second_sq = torch.zeros_like(first_sq)
            dot = torch.zeros_like(first_sq)
            for first_grad, second_grad in zip(first, second, strict=True):
                if first_grad is not None:
                    first_sq += first_grad.detach().float().square().sum()
                if second_grad is not None:
                    second_sq += second_grad.detach().float().square().sum()
                if first_grad is not None and second_grad is not None:
                    dot += (first_grad.detach().float() * second_grad.detach().float()).sum()
            first_norm = torch.sqrt(first_sq)
            second_norm = torch.sqrt(second_sq)
            cosine = dot / (first_norm * second_norm).clamp_min(1.0e-12)
            # Magnitude of the first gradient that directly opposes the unit
            # Tracking-Q direction. Zero means it is neutral or cooperative.
            conflict_projection = torch.clamp(-dot / second_norm.clamp_min(1.0e-12), min=0.0)
            return first_norm, cosine, conflict_projection

        tracking_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for gradient in tracking_gradients:
            if gradient is not None:
                tracking_sq += gradient.detach().float().square().sum()
        tracking_norm = torch.sqrt(tracking_sq)
        base_norm, base_cosine, _ = compare(base_gradients, tracking_gradients)
        names = tuple(component_losses)
        values = [base_norm, tracking_norm, base_cosine]
        for name in names:
            values.extend(compare(component_gradients[name], tracking_gradients))
        statistics = torch.stack(values)
        if is_distributed():
            import torch.distributed as dist

            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            statistics /= float(dist.get_world_size())

        base_value, tracking_value, base_cosine_value = statistics[:3].tolist()
        self._tracking_grad_cosine = float(base_cosine_value)
        if base_value > 1.0e-12 and tracking_value > 1.0e-12:
            beta = float(self.cfg.train.tracking_grad_ema_beta)
            initialized = self._tracking_grad_ema_initialized
            if initialized:
                self._tracking_grad_base_ema = beta * self._tracking_grad_base_ema + (1.0 - beta) * base_value
                self._tracking_grad_q_ema = beta * self._tracking_grad_q_ema + (1.0 - beta) * tracking_value
            else:
                self._tracking_grad_base_ema = float(base_value)
                self._tracking_grad_q_ema = float(tracking_value)
                self._tracking_grad_ema_initialized = True
            target = self._tracking_grad_base_ema / max(self._tracking_grad_q_ema, 1.0e-12)
            target = min(max(target, self.cfg.train.tracking_grad_scale_min), self.cfg.train.tracking_grad_scale_max)
            if initialized:
                limit = float(self.cfg.train.tracking_grad_scale_rate_limit)
                target = min(max(target, self._tracking_grad_scale / limit), self._tracking_grad_scale * limit)
            self._tracking_grad_scale = float(target)

        diagnostics = {"tracking_norm": float(tracking_value)}
        cursor = 3
        for name in names:
            norm, cosine, conflict_projection = statistics[cursor : cursor + 3].tolist()
            diagnostics[f"{name}/norm"] = float(norm)
            diagnostics[f"{name}/cosine"] = float(cosine)
            diagnostics[f"{name}/conflict_projection"] = float(conflict_projection)
            cursor += 3
        self._tracking_component_grad_metrics = diagnostics

    def _tracking_gradient_output_metrics(
        self,
        *,
        component_coefficients: dict[str, float],
        tracking_coefficient: float,
    ) -> dict[str, torch.Tensor]:
        diagnostics = self._tracking_component_grad_metrics
        output = {
            "tracking_q/grad_norm_tracking_raw": torch.tensor(
                diagnostics.get("tracking_norm", 0.0), device=self.device
            ),
            "tracking_q/grad_norm_tracking_weighted": torch.tensor(
                abs(float(tracking_coefficient)) * diagnostics.get("tracking_norm", 0.0),
                device=self.device,
            ),
        }
        for name, coefficient in component_coefficients.items():
            coefficient = float(coefficient)
            output[f"tracking_q/grad_norm_{name}"] = torch.tensor(
                abs(coefficient) * diagnostics.get(f"{name}/norm", 0.0), device=self.device
            )
            output[f"tracking_q/grad_cosine_{name}_vs_tracking"] = torch.tensor(
                diagnostics.get(f"{name}/cosine", 0.0), device=self.device
            )
            output[f"tracking_q/conflict_projection_{name}_vs_tracking"] = torch.tensor(
                max(coefficient, 0.0) * diagnostics.get(f"{name}/conflict_projection", 0.0),
                device=self.device,
            )
        return output

    @staticmethod
    def _gradient_statistics(
        acquisition_loss: torch.Tensor,
        consolidation_loss: torch.Tensor,
        parameters,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parameters = tuple(parameters)
        acquisition_gradients = torch.autograd.grad(
            acquisition_loss, parameters, retain_graph=True, allow_unused=True
        )
        consolidation_gradients = torch.autograd.grad(
            consolidation_loss, parameters, retain_graph=True, allow_unused=True
        )
        acquisition_sq = torch.zeros((), device=acquisition_loss.device, dtype=torch.float32)
        consolidation_sq = torch.zeros_like(acquisition_sq)
        dot = torch.zeros_like(acquisition_sq)
        for acquisition_gradient, consolidation_gradient in zip(
            acquisition_gradients, consolidation_gradients, strict=True
        ):
            if acquisition_gradient is not None:
                acquisition_sq += acquisition_gradient.detach().float().square().sum()
            if consolidation_gradient is not None:
                consolidation_sq += consolidation_gradient.detach().float().square().sum()
            if acquisition_gradient is not None and consolidation_gradient is not None:
                dot += (
                    acquisition_gradient.detach().float() * consolidation_gradient.detach().float()
                ).sum()
        acquisition_norm = torch.sqrt(acquisition_sq)
        consolidation_norm = torch.sqrt(consolidation_sq)
        cosine = dot / (acquisition_norm * consolidation_norm).clamp_min(1.0e-12)
        statistics = torch.stack((acquisition_norm, consolidation_norm, cosine))
        if is_distributed():
            import torch.distributed as dist

            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            statistics /= float(dist.get_world_size())
        return statistics.unbind()

    def _update_pace_gradient_scale(
        self,
        acquisition_loss: torch.Tensor,
        consolidation_loss: torch.Tensor,
    ) -> None:
        acquisition_norm, consolidation_norm, cosine = self._gradient_statistics(
            acquisition_loss,
            consolidation_loss,
            self._model._actor.parameters(),
        )
        acquisition_value = float(acquisition_norm.item())
        consolidation_value = float(consolidation_norm.item())
        # At the exact base policy L_con and its gradient are both zero. That
        # is not evidence for an infinite calibration scale; wait until the
        # first small acquisition update creates a measurable displacement.
        if acquisition_value <= 1.0e-12 or consolidation_value <= 1.0e-12:
            self._pace_grad_cosine = float(cosine.item())
            return
        beta = self._pace_gradient_scale_ema_beta
        was_initialized = self._pace_grad_ema_initialized
        if was_initialized:
            self._pace_grad_acquisition_ema = (
                beta * self._pace_grad_acquisition_ema + (1.0 - beta) * acquisition_value
            )
            self._pace_grad_consolidation_ema = (
                beta * self._pace_grad_consolidation_ema + (1.0 - beta) * consolidation_value
            )
        else:
            self._pace_grad_acquisition_ema = acquisition_value
            self._pace_grad_consolidation_ema = consolidation_value
            self._pace_grad_ema_initialized = True
        target_scale = self._pace_grad_acquisition_ema / max(self._pace_grad_consolidation_ema, 1.0e-12)
        target_scale = min(max(target_scale, self._pace_gradient_scale_min), self._pace_gradient_scale_max)
        if was_initialized:
            rate_limit = self._pace_gradient_scale_rate_limit
            lower = self._pace_consolidation_scale / rate_limit
            upper = self._pace_consolidation_scale * rate_limit
            self._pace_consolidation_scale = min(max(target_scale, lower), upper)
        else:
            self._pace_consolidation_scale = target_scale
        self._pace_grad_cosine = float(cosine.item())

    def update_actor_pace(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        acquisition_mask: torch.Tensor,
        clip_grad_norm: float | None,
        tracking_obs: dict[str, torch.Tensor] | None = None,
        tracking_z: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self._pace_reference_actor is None:
            raise RuntimeError("PACE reference actor has not been initialized")
        acquisition_mask = acquisition_mask.reshape(-1).bool()
        consolidation_mask = ~acquisition_mask

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            sampled_action = dist.sample(clip=self.cfg.train.stddev_clip)
            if self.cfg.train.disable_discriminator:
                q_discriminator = None
            else:
                qs_discriminator = self._model._critic(obs, z, sampled_action)
                _, _, q_discriminator = self.get_targets_uncertainty(
                    qs_discriminator, self.cfg.train.actor_pessimism_penalty
                )
            qs_aux = self._model._aux_critic(obs, z, sampled_action)
            _, _, q_aux = self.get_targets_uncertainty(qs_aux, self.cfg.train.actor_pessimism_penalty)
            fs = self._model._forward_map(obs, z, sampled_action)
            qs_fb = (fs * z).sum(-1)
            _, _, q_fb = self.get_targets_uncertainty(qs_fb, self.cfg.train.actor_pessimism_penalty)

            zero = dist.mean.sum() * 0.0
            if bool(acquisition_mask.any()):
                acq_q_aux = q_aux[acquisition_mask]
                acq_q_fb = q_fb[acquisition_mask]
                weight = acq_q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
                auxiliary_actor_loss = -acq_q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
                fb_actor_loss = -acq_q_fb.mean()
                if q_discriminator is None:
                    acq_q_discriminator = zero.reshape(1)
                    discriminator_actor_loss = None
                    base_acquisition_loss = auxiliary_actor_loss + fb_actor_loss
                else:
                    acq_q_discriminator = q_discriminator[acquisition_mask]
                    discriminator_actor_loss = -acq_q_discriminator.mean() * self.cfg.train.reg_coeff * weight
                    base_acquisition_loss = discriminator_actor_loss + auxiliary_actor_loss + fb_actor_loss
            else:
                acq_q_discriminator = zero.reshape(1)
                acq_q_aux = zero.reshape(1)
                acq_q_fb = zero.reshape(1)
                discriminator_actor_loss = None if q_discriminator is None else zero
                auxiliary_actor_loss = zero
                fb_actor_loss = zero
                base_acquisition_loss = zero

            if tracking_obs is not None and tracking_z is not None:
                tracking_dist = self._model._actor(tracking_obs, tracking_z, self._model.cfg.actor_std)
                tracking_action = tracking_dist.sample(clip=self.cfg.train.stddev_clip)
                tracking_qs = self._model._tracking_critic(tracking_obs, tracking_z, tracking_action)
                _, tracking_uncertainty, tracking_q = self.get_targets_uncertainty(
                    tracking_qs,
                    self.cfg.train.tracking_critic_pessimism_penalty,
                )
                tracking_actor_loss = -tracking_q.mean()
            else:
                tracking_q = zero.reshape(1)
                tracking_uncertainty = tracking_q
                tracking_actor_loss = zero

            if bool(consolidation_mask.any()):
                consolidation_obs = tree_map(lambda value: value[consolidation_mask], obs)
                consolidation_z = z[consolidation_mask]
                with torch.no_grad():
                    reference_mean = self._pace_reference_actor(
                        consolidation_obs,
                        consolidation_z,
                        self._model.cfg.actor_std,
                    ).mean
                current_mean = dist.mean[consolidation_mask]
                consolidation_loss = (current_mean - reference_mean).square().sum(dim=-1).mean()
            else:
                consolidation_loss = zero
        self._tracking_update_count += 1
        if tracking_obs is not None:
            tracking_monitor_every = int(self.cfg.train.tracking_grad_monitor_every_updates)
            if self._tracking_update_count >= int(self.cfg.train.tracking_actor_warmup_updates) and (
                not self._tracking_grad_ema_initialized
                or self._tracking_update_count % tracking_monitor_every == 0
            ):
                component_losses = {}
                if discriminator_actor_loss is not None:
                    component_losses["discriminator"] = discriminator_actor_loss
                component_losses["auxiliary"] = auxiliary_actor_loss
                component_losses["fb"] = fb_actor_loss
                component_losses["consolidation"] = consolidation_loss
                self._update_tracking_gradient_diagnostics(
                    base_loss=base_acquisition_loss,
                    component_losses=component_losses,
                    tracking_loss=tracking_actor_loss,
                )
        tracking_coefficient = self._tracking_actor_coefficient() if tracking_obs is not None else 0.0
        acquisition_loss = base_acquisition_loss + tracking_coefficient * tracking_actor_loss

        self._pace_update_count += 1
        monitor_every = int(getattr(self, "_pace_grad_monitor_every_updates", 100))
        should_measure = self._pace_actor_updates_enabled and (
            not self._pace_grad_ema_initialized or self._pace_update_count % monitor_every == 0
        )
        # All distributed ranks must enter the calibration collective in the
        # same order even if a rare local batch contains only one role.
        if should_measure:
            self._update_pace_gradient_scale(acquisition_loss, consolidation_loss)

        effective_consolidation_coefficient = self._pace_lambda_con * self._pace_consolidation_scale
        weighted_acquisition_loss = self._pace_acquisition_loss_weight * acquisition_loss
        weighted_consolidation_loss = effective_consolidation_coefficient * consolidation_loss
        actor_loss = weighted_acquisition_loss + weighted_consolidation_loss
        apply_actor_update = (
            self._pace_actor_updates_enabled
            and self._pace_update_count % self._pace_actor_update_every == 0
        )
        if apply_actor_update:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            average_gradients(self._model._actor.parameters())
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
            self.actor_optimizer.step()

        grad_acquisition = torch.tensor(self._pace_grad_acquisition_ema, device=self.device)
        grad_consolidation = torch.tensor(self._pace_grad_consolidation_ema, device=self.device)
        weighted_grad_consolidation = effective_consolidation_coefficient * grad_consolidation
        weighted_grad_ratio = weighted_grad_consolidation / grad_acquisition.clamp_min(1.0e-12)
        current_actor_lr = float(self.actor_optimizer.param_groups[0]["lr"])

        output_metrics = {
            "actor_loss": actor_loss.detach(),
            "Q_discriminator": acq_q_discriminator.mean().detach(),
            "Q_aux": acq_q_aux.mean().detach(),
            "Q_fb": acq_q_fb.mean().detach(),
            "pace/loss_acquisition": acquisition_loss.detach(),
            "pace/loss_acquisition_base": base_acquisition_loss.detach(),
            "pace/loss_acquisition_weighted": weighted_acquisition_loss.detach(),
            "pace/loss_consolidation": consolidation_loss.detach(),
            "pace/loss_consolidation_weighted": weighted_consolidation_loss.detach(),
            "pace/actor_grad_norm_acquisition": grad_acquisition,
            "pace/actor_grad_norm_consolidation": grad_consolidation,
            "pace/actor_grad_norm_consolidation_weighted": weighted_grad_consolidation,
            "pace/actor_grad_ratio_consolidation_to_acquisition": weighted_grad_ratio,
            "pace/actor_grad_cosine": torch.tensor(self._pace_grad_cosine, device=self.device),
            "pace/consolidation_scale": torch.tensor(self._pace_consolidation_scale, device=self.device),
            "pace/effective_consolidation_coefficient": torch.tensor(
                effective_consolidation_coefficient, device=self.device
            ),
            "pace/consolidation_action_rmse": torch.sqrt(
                consolidation_loss.detach() / max(int(self.action_dim), 1)
            ),
            "pace/actor_update_applied": torch.tensor(float(apply_actor_update), device=self.device),
            "pace/actor_lr": torch.tensor(current_actor_lr, device=self.device),
            "pace/acquisition_loss_weight": torch.tensor(
                self._pace_acquisition_loss_weight, device=self.device
            ),
            "pace/batch_acquisition": acquisition_mask.sum().detach(),
            "pace/batch_consolidation": consolidation_mask.sum().detach(),
            "tracking_q/actor_q": tracking_q.mean().detach(),
            "tracking_q/actor_uncertainty": tracking_uncertainty.mean().detach(),
            "tracking_q/actor_loss": tracking_actor_loss.detach(),
            "tracking_q/base_actor_loss": base_acquisition_loss.detach(),
            "tracking_q/actor_coefficient": torch.tensor(tracking_coefficient, device=self.device),
            "tracking_q/grad_scale": torch.tensor(self._tracking_grad_scale, device=self.device),
            "tracking_q/grad_base_ema": torch.tensor(self._tracking_grad_base_ema, device=self.device),
            "tracking_q/grad_tracking_ema": torch.tensor(self._tracking_grad_q_ema, device=self.device),
            "tracking_q/weighted_grad_ratio": torch.tensor(
                tracking_coefficient * self._tracking_grad_q_ema / max(self._tracking_grad_base_ema, 1.0e-12),
                device=self.device,
            ),
            "tracking_q/grad_cosine": torch.tensor(self._tracking_grad_cosine, device=self.device),
            "tracking_q/actor_ramp": torch.tensor(
                tracking_coefficient
                / max(self.cfg.train.tracking_actor_grad_ratio * self._tracking_grad_scale, 1.0e-12),
                device=self.device,
            ),
        }
        component_coefficients = {}
        if discriminator_actor_loss is not None:
            component_coefficients["discriminator"] = self._pace_acquisition_loss_weight
        component_coefficients["auxiliary"] = self._pace_acquisition_loss_weight
        component_coefficients["fb"] = self._pace_acquisition_loss_weight
        component_coefficients["consolidation"] = effective_consolidation_coefficient
        output_metrics.update(
            self._tracking_gradient_output_metrics(
                component_coefficients=component_coefficients,
                tracking_coefficient=self._pace_acquisition_loss_weight * tracking_coefficient,
            )
        )
        return output_metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            Qs = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        average_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_tracking_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        tracking_reward: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        z: torch.Tensor,
        next_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.tracking_critic_optimizer is None:
            raise RuntimeError("Tracking critic optimizer is not initialized")
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = int(self.cfg.model.archi.tracking_critic.num_parallel)
            with torch.no_grad():
                next_dist = self._model._actor(next_obs, next_z, self._model.cfg.actor_std)
                next_action = next_dist.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_tracking_critic(next_obs, next_z, next_action)
                next_q_mean, next_q_uncertainty, next_value = self.get_targets_uncertainty(
                    next_qs,
                    self.cfg.train.tracking_critic_pessimism_penalty,
                )
                target_q = tracking_reward + discount * next_value
                expanded_target = target_q.expand(num_parallel, -1, -1)
            qs = self._model._tracking_critic(obs, z, action)
            tracking_critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_target)

        self.tracking_critic_optimizer.zero_grad(set_to_none=True)
        tracking_critic_loss.backward()
        average_gradients(self._model._tracking_critic.parameters())
        self.tracking_critic_optimizer.step()
        return {
            "tracking_q/target": target_q.mean().detach(),
            "tracking_q/value": qs.mean().detach(),
            "tracking_q/next_value": next_q_mean.mean().detach(),
            "tracking_q/uncertainty": next_q_uncertainty.mean().detach(),
            "tracking_q/critic_loss": tracking_critic_loss.detach(),
            "tracking_q/reward": tracking_reward.mean().detach(),
        }

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
        tracking_obs: dict[str, torch.Tensor] | None = None,
        tracking_z: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            # compute discriminator reward loss (skipped in no-discriminator mode)
            if self.cfg.train.disable_discriminator:
                Q_discriminator = None
            else:
                Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
                _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute auxiliary reward loss
            Qs_aux = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            auxiliary_actor_loss = -Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
            fb_actor_loss = -Q_fb.mean()
            if Q_discriminator is None:
                Q_discriminator = Q_fb.detach() * 0.0
                discriminator_actor_loss = None
                base_actor_loss = auxiliary_actor_loss + fb_actor_loss
            else:
                discriminator_actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
                base_actor_loss = discriminator_actor_loss + auxiliary_actor_loss + fb_actor_loss
            if tracking_obs is not None and tracking_z is not None:
                tracking_dist = self._model._actor(tracking_obs, tracking_z, self._model.cfg.actor_std)
                tracking_action = tracking_dist.sample(clip=self.cfg.train.stddev_clip)
                tracking_qs = self._model._tracking_critic(tracking_obs, tracking_z, tracking_action)
                _, tracking_uncertainty, tracking_q = self.get_targets_uncertainty(
                    tracking_qs,
                    self.cfg.train.tracking_critic_pessimism_penalty,
                )
                tracking_actor_loss = -tracking_q.mean()
            else:
                tracking_q = base_actor_loss.detach().reshape(1) * 0.0
                tracking_uncertainty = tracking_q
                tracking_actor_loss = base_actor_loss * 0.0

        self._tracking_update_count += 1
        if tracking_obs is not None:
            monitor_every = int(self.cfg.train.tracking_grad_monitor_every_updates)
            should_monitor = (
                self._tracking_update_count >= int(self.cfg.train.tracking_actor_warmup_updates)
                and (
                    not self._tracking_grad_ema_initialized
                    or self._tracking_update_count % monitor_every == 0
                )
            )
            if should_monitor:
                component_losses = {}
                if discriminator_actor_loss is not None:
                    component_losses["discriminator"] = discriminator_actor_loss
                component_losses["auxiliary"] = auxiliary_actor_loss
                component_losses["fb"] = fb_actor_loss
                self._update_tracking_gradient_diagnostics(
                    base_loss=base_actor_loss,
                    component_losses=component_losses,
                    tracking_loss=tracking_actor_loss,
                )
        tracking_coefficient = self._tracking_actor_coefficient() if tracking_obs is not None else 0.0
        actor_loss = base_actor_loss + tracking_coefficient * tracking_actor_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "tracking_q/actor_q": tracking_q.mean().detach(),
                "tracking_q/actor_uncertainty": tracking_uncertainty.mean().detach(),
                "tracking_q/actor_loss": tracking_actor_loss.detach(),
                "tracking_q/base_actor_loss": base_actor_loss.detach(),
                "tracking_q/actor_coefficient": torch.tensor(tracking_coefficient, device=self.device),
                "tracking_q/grad_scale": torch.tensor(self._tracking_grad_scale, device=self.device),
                "tracking_q/grad_base_ema": torch.tensor(self._tracking_grad_base_ema, device=self.device),
                "tracking_q/grad_tracking_ema": torch.tensor(self._tracking_grad_q_ema, device=self.device),
                "tracking_q/weighted_grad_ratio": torch.tensor(
                    tracking_coefficient * self._tracking_grad_q_ema / max(self._tracking_grad_base_ema, 1.0e-12),
                    device=self.device,
                ),
                "tracking_q/grad_cosine": torch.tensor(self._tracking_grad_cosine, device=self.device),
                "tracking_q/actor_ramp": torch.tensor(
                    tracking_coefficient
                    / max(self.cfg.train.tracking_actor_grad_ratio * self._tracking_grad_scale, 1.0e-12),
                    device=self.device,
                ),
            }
            component_coefficients = {}
            if discriminator_actor_loss is not None:
                component_coefficients["discriminator"] = 1.0
            component_coefficients["auxiliary"] = 1.0
            component_coefficients["fb"] = 1.0
            output_metrics.update(
                self._tracking_gradient_output_metrics(
                    component_coefficients=component_coefficients,
                    tracking_coefficient=tracking_coefficient,
                )
            )
        return output_metrics
