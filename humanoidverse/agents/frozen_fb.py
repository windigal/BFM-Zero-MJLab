"""Load and freeze a pretrained FB representation for PPO-FB.

PPO-FB keeps the pretrained ``B`` (and the observation normalizer that feeds it) fixed and
trains a new controller against rewards derived from it. This module only handles the
representation: resolving a checkpoint, loading ``B``/normalizer weights into a model, and
freezing them. It deliberately does not touch ``F``, the actor or any critic.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

REPRESENTATION_PREFIXES = ("_backward_map.", "_obs_normalizer.")


def resolve_fb_checkpoint(path: str | Path) -> Path:
    """Accept a run dir or its ``checkpoint`` dir; return the dir holding ``model/model.safetensors``."""
    root = Path(path).expanduser().resolve()
    for candidate in (root, root / "checkpoint"):
        if (candidate / "model" / "model.safetensors").exists():
            if not (candidate / "model" / "config.json").exists():
                raise FileNotFoundError(f"FB checkpoint is incomplete; missing {candidate / 'model' / 'config.json'}")
            return candidate
    raise FileNotFoundError(f"No model/model.safetensors found under {root} or {root / 'checkpoint'}")


def load_representation_state(checkpoint: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return only the ``B`` and observation-normalizer tensors from a pretrained model."""
    model_path = resolve_fb_checkpoint(checkpoint) / "model" / "model.safetensors"
    selected = load_checkpoint_tensors(checkpoint, REPRESENTATION_PREFIXES, device)
    if not any(k.startswith("_backward_map.") for k in selected):
        raise RuntimeError(f"{model_path} has no '_backward_map.*' tensors; is it an FB model?")
    return selected


def load_checkpoint_tensors(checkpoint, prefixes: tuple[str, ...], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read selected tensors without materializing the unused forward maps/critics."""
    path = resolve_fb_checkpoint(checkpoint) / "model" / "model.safetensors"
    with safe_open(path, framework="pt", device=str(device)) as source:
        return {key: source.get_tensor(key) for key in source.keys() if key.startswith(prefixes)}


def pretrained_z_dim(checkpoint: str | Path) -> int:
    with (resolve_fb_checkpoint(checkpoint) / "model" / "config.json").open() as f:
        return int(json.load(f)["archi"]["z_dim"])


def freeze_representation(model: nn.Module) -> None:
    """Put ``B`` (and its target copy) and the normalizer in eval mode with gradients off."""
    for name in ("_backward_map", "_target_backward_map", "_obs_normalizer"):
        module = getattr(model, name, None)
        if module is not None:
            module.eval().requires_grad_(False)


def load_frozen_representation(model: nn.Module, checkpoint: str | Path, device: str = "cpu") -> list[str]:
    """Copy pretrained ``B``/normalizer weights into ``model`` strictly, then freeze them.

    Raises if any tensor is missing, unexpected or shape-mismatched, so a pretrained model
    with a different observation space or ``z_dim`` fails loudly instead of loading partially.
    Returns the names of the loaded tensors.
    """
    pretrained = load_representation_state(checkpoint, device=device)
    expected = {k: v for k, v in model.state_dict().items() if k.startswith(REPRESENTATION_PREFIXES)}
    missing = sorted(set(expected) - set(pretrained))
    unexpected = sorted(set(pretrained) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(pretrained) if tuple(expected[k].shape) != tuple(pretrained[k].shape))
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Pretrained representation does not match the current model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )
    model.load_state_dict(pretrained, strict=False)
    # The target copy (when the model was prepared for training) must start from the same B.
    target = getattr(model, "_target_backward_map", None)
    if target is not None:
        target.load_state_dict(model._backward_map.state_dict())
    freeze_representation(model)
    return sorted(pretrained)


@torch.no_grad()
def encode_expert_latents(
    model,
    observations,
    *,
    seq_length: int,
    batch_size: int = 32768,
) -> torch.Tensor:
    """Per-step tracking latents ``z_t`` for one expert motion, in a single pass.

    ``observations`` are the motion's next-state observations (steps ``1..L``). ``z_t`` is the
    projected mean of ``B`` over ``observations[t : t + seq_length]``, shortened near the end of
    the motion. This is the same quantity as ``model.tracking_inference`` computed with prefix
    sums instead of a Python loop, so a whole motion costs one ``B`` pass.
    """
    from torch.utils._pytree import tree_map

    first = next(iter(observations.values()))
    total = first.shape[0]
    if total == 0 or seq_length < 1 or batch_size < 1:
        raise ValueError("Expert observations, seq_length and batch_size must be nonempty/positive")
    parts = []
    for begin in range(0, total, batch_size):
        chunk = tree_map(lambda value: value[begin : begin + batch_size].to(model.device), observations)
        parts.append(model.backward_map(chunk))
    embeddings = torch.cat(parts, dim=0).float()
    prefix = torch.cat((torch.zeros_like(embeddings[:1]), embeddings.cumsum(dim=0)), dim=0)
    starts = torch.arange(total, device=embeddings.device)
    ends = torch.clamp(starts + int(seq_length), max=total)
    rolling = (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(-1)
    return model.project_z(rolling)
