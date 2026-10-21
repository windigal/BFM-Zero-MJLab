"""Inspect selective FB weights, expert second moments, and optional tracking baselines."""

import argparse
import json
from pathlib import Path

import torch

from humanoidverse.ppo_fb.pretrained import PretrainedController
from humanoidverse.ppo_fb.runner import evaluate_tracking
from humanoidverse.ppo_fb.train import build_environment, build_expert_buffer, build_parser


@torch.no_grad()
def second_moment(model, observations, samples=65536, batch_size=8192):
    count = next(iter(observations.values())).shape[0]
    indices = torch.randint(count, (samples,))
    moment = torch.zeros(model.cfg.archi.z_dim, model.cfg.archi.z_dim, dtype=torch.float64)
    for begin in range(0, samples, batch_size):
        batch = {key: value[indices[begin : begin + batch_size]].to(model.device) for key, value in observations.items()}
        embedding = model.backward_map(batch).double().cpu()
        moment += embedding.T @ embedding
    return moment / samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "runs/pretrained/20260804_040010/checkpoint",
            "runs/pretrained/20260810_002844/checkpoint",
        ],
    )
    parser.add_argument("--data-path", default="humanoidverse/data/seed_train_full10s_5k.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-motions", type=int, help="Limit FK preprocessing; omitted means the whole dataset")
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--save-inverse", help="Save the first checkpoint's frozen inverse Sigma for whitening ablation")
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--output", help="JSON report path; omitted means stdout only")
    args = parser.parse_args()
    if args.samples < 1 or not 0 <= args.shrinkage <= 1:
        parser.error("samples must be positive and shrinkage must be in [0, 1]")
    torch.set_num_threads(min(torch.get_num_threads(), 12))
    training = build_parser().parse_args([])
    for key in ("device", "num_envs", "data_path", "seed"):
        setattr(training, key, getattr(args, key))
    torch.manual_seed(args.seed)
    first = PretrainedController(args.checkpoints[0], args.device)
    buffer = build_expert_buffer(training, first, args.max_motions)
    observations = {key: value for key, value in buffer.storage["observation"].items() if key in ("state", "privileged_state")}
    report = {"motion_count": len(buffer.motion_ids), "expert_frames": len(buffer), "checkpoints": {}}
    del buffer
    for index, checkpoint in enumerate(args.checkpoints):
        model = first if index == 0 else PretrainedController(checkpoint, args.device)
        torch.manual_seed(args.seed)  # identical sampled expert states for both checkpoints
        moment = second_moment(model, observations, args.samples)
        eigenvalues = torch.linalg.eigvalsh(moment)
        components = {}
        for name in ("_backward_map", "_obs_normalizer", "_actor", "_discriminator"):
            module = getattr(model, name)
            components[name] = {
                "parameters": sum(value.numel() for value in module.parameters()),
                "state_mib": sum(value.numel() * value.element_size() for value in module.state_dict().values()) / 2**20,
            }
        result = {
            "components": components,
            "sigma": {
                "samples": args.samples,
                "trace": moment.trace().item(),
                "min_eigenvalue": eigenvalues[0].item(),
                "max_eigenvalue": eigenvalues[-1].item(),
                "condition_number": (eigenvalues[-1] / eigenvalues[0]).item(),
                "eigenvalues": eigenvalues.tolist(),
            },
        }
        if index == 0 and args.save_inverse:
            shrunk = (1 - args.shrinkage) * moment + args.shrinkage * torch.eye(moment.shape[0])
            torch.save(torch.linalg.inv(shrunk).float(), args.save_inverse)
        if args.tracking:
            env = build_environment(training)
            try:
                result["tracking"] = evaluate_tracking(env, model, max_motions=args.max_motions, selection_seed=args.seed)
            finally:
                env.close()
        report["checkpoints"][checkpoint] = result
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
