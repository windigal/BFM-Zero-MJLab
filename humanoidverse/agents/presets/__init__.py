"""Agent presets used by the UFO training entrypoint."""

from __future__ import annotations

from typing import Any

from humanoidverse.agents.presets.fb import TRAIN_RUNTIME as FB_TRAIN_RUNTIME
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.agents.presets.tldr import TRAIN_RUNTIME as TECH_TRAIN_RUNTIME
from humanoidverse.agents.presets.tldr import build_tldr_agent
from humanoidverse.agents.presets.tldr import build_tldr_agent as build_tech_agent


def build_agent_preset(
    *,
    agent: str,
    device: str,
    compile: bool,
    update_z_every_step: int,
    lr_scale: float,
    clip_grad_norm: float,
    cartwheel_aux_safe: bool,
    enable_tracking_q: bool = False,
    freeze_backward: bool = False,
    tracking_actor_grad_ratio: float = 0.10,
    disable_discriminator: bool = False,
    wandb_project: str,
    wandb_group: str = "mjlab",
) -> dict[str, Any]:
    if agent == "fb":
        return {
            "agent_cfg": build_fb_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
                enable_tracking_q=enable_tracking_q,
                freeze_backward=freeze_backward,
                tracking_actor_grad_ratio=tracking_actor_grad_ratio,
                disable_discriminator=disable_discriminator,
            ),
            "wandb_group": wandb_group,
            "wandb_project": wandb_project,
            "train_runtime": dict(FB_TRAIN_RUNTIME),
        }
    if agent in {"tech", "tldr"}:
        if enable_tracking_q or freeze_backward or disable_discriminator:
            raise ValueError("Extreme-RGMT tracking Q is currently supported only by the fb preset")
        return {
            "agent_cfg": build_tech_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
            ),
            "wandb_group": wandb_group,
            "wandb_project": wandb_project,
            "train_runtime": dict(TECH_TRAIN_RUNTIME),
        }
    raise ValueError(f"Unsupported agent preset: {agent}")


__all__ = ["build_agent_preset", "build_fb_agent", "build_tech_agent", "build_tldr_agent"]
