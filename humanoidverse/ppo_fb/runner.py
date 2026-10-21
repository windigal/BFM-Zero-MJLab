"""Reuse rsl-rl's rollout loop, with multi-head logging and tracking evaluation."""

import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig

from .pretrained import EvaluationAgent
from .storage import MultiHeadRolloutStorage


class _BudgetReached(Exception):
    pass


class _WarmupFailed(RuntimeError):
    pass


class PpoFbLogger(Logger):
    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        weights = self.reward_weights if rewards.shape[-1] == 4 else rewards.new_ones(1)
        super().process_env_step((rewards * weights).sum(-1), dones, extras, intrinsic_rewards)

    def log(self, **kwargs):
        if self.writer is None:
            self.tot_timesteps += self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        super().log(**kwargs)
        self.iteration_hook(kwargs["it"] + 1)


@torch.inference_mode()
def evaluate_tracking(env, pretrained, actor=None, *, max_motions=None, selection_seed=0, motion_pool_size=1024):
    evaluator = HumanoidVerseMjlabTrackingEvaluationConfig(
        num_envs=env.num_envs,
        # Match 0804's logged EMD: its evaluator used one next-state embedding.
        seq_length=1,
        max_motions=max_motions,
        motion_batch_size=motion_pool_size,
        selection_seed=selection_seed,
        body_metrics=True,
    ).build()
    # The evaluator logs the returned metrics; the runner writes them at its own iteration.
    reward_model = env.base_env.ppo_reward_model
    env.base_env.ppo_reward_model = None
    try:
        _, aggregate = evaluator.run(timestep=0, agent_or_model=EvaluationAgent(pretrained, actor), logger=None, env=env)
    finally:
        env.base_env.ppo_reward_model = reward_model
    return aggregate


class PpoFbRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = PpoFbLogger(
            log_dir,
            self.cfg,
            env.cfg,
            env.num_envs,
            self.is_distributed,
            self.gpu_world_size,
            self.gpu_global_rank,
            self.device,
        )
        self.logger.reward_weights = env.reward_weights.to(device)
        self.logger.iteration_hook = self._iteration_end
        self.logger.writer = None
        self.pool_rng = np.random.default_rng(self.cfg["seed"])
        self.pool_order = self.pool_rng.permutation(len(env.experts.lengths))
        self.pool_cursor, self.pool_steps = 0, 0
        self.training_motion_ids = env.core._motion_lib._curr_motion_ids.cpu().tolist()
        self._configure_rollout_length()
        self.add_git_repo_to_log(__file__)

    def _configure_rollout_length(self):
        steps = self.cfg["warmup_rollout_steps"] if self.alg.warming_up else self.cfg["joint_rollout_steps"]
        if steps != self.alg.storage.num_transitions_per_env:
            self.alg.storage = MultiHeadRolloutStorage(
                self.env.num_envs,
                steps,
                self.env.get_observations(),
                [self.env.num_actions],
                self.alg.storage.num_heads,
                self.device,
            )
        self.cfg["num_steps_per_env"] = steps

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Random episode lengths would break expert/reference alignment")
        self.current_learning_iteration = self.alg.iteration
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len=False)
        except (_BudgetReached, _WarmupFailed) as error:
            # Both budget exhaustion and failed warmup retain a resumable model.
            if self.logger.writer is not None:
                self.save(f"{self.logger.log_dir}/model_{self.alg.iteration}.pt")
                self.logger.stop_logging_writer()
            if isinstance(error, _WarmupFailed):
                raise

    def _advance_motion_pool(self):
        if self.pool_cursor == len(self.pool_order):
            self.pool_order = self.pool_rng.permutation(len(self.pool_order))
            self.pool_cursor = 0
        end = min(self.pool_cursor + self.cfg["motion_pool_size"], len(self.pool_order))
        self.training_motion_ids = self.pool_order[self.pool_cursor : end].tolist()
        self.pool_cursor = end
        self.pool_steps = 0

    def _iteration_end(self, iteration):
        self.pool_steps += self.cfg["num_steps_per_env"]
        if self.alg.warming_up and iteration >= self.alg.warmup_max_iterations:
            raise _WarmupFailed("Critic warmup did not converge; actor remains frozen. Inspect validation metrics before resuming.")
        if self.logger.tot_timesteps >= self.cfg["max_env_steps"]:
            raise _BudgetReached
        refresh = self.pool_steps >= self.cfg["motion_pool_refresh_steps"]
        if refresh:
            self._advance_motion_pool()
        interval = self.cfg["eval_interval"]
        evaluate = bool(interval and iteration % interval == 0)
        if evaluate or refresh:
            self.alg.eval_mode()
            try:
                if evaluate:
                    metrics = evaluate_tracking(
                        self.env.env,
                        self.env.pretrained,
                        self.alg.actor,
                        max_motions=self.cfg["eval_motions"],
                        selection_seed=self.cfg["eval_seed"],
                        motion_pool_size=self.cfg["motion_pool_size"],
                    )
                    if self.logger.writer is not None:
                        for name, value in metrics.items():
                            self.logger.writer.add_scalar(f"Eval/{name}", value, iteration - 1)
            finally:
                self.env.core.prepare_training_motion_pool(motion_ids=self.training_motion_ids)
                with torch.inference_mode():
                    # The base learn loop holds this TensorDict across iterations.
                    # Refresh that same object after evaluation resets the vector env.
                    self.alg.last_observations.copy_(self.env.reset())
                self.alg.train_mode()
        self._configure_rollout_length()

    def save(self, path, infos=None):
        saved = self.alg.save() | {
            "iter": self.alg.iteration,
            "infos": infos,
            "config": self.cfg,
            "frozen_reward_state_dict": self.env.core.ppo_reward_model.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "total_timesteps": self.logger.tot_timesteps,
            "motion_pool": {
                "order": self.pool_order,
                "cursor": self.pool_cursor,
                "steps": self.pool_steps,
                "ids": self.training_motion_ids,
                "rng": self.pool_rng.bit_generator.state,
                "seen_expert_motions": self.env.seen_expert_motions,
            },
        }
        if torch.cuda.is_available():
            saved["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(saved, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, weights_only=False, map_location=map_location or self.device)
        # Cached expert z must use exactly the same representation as the reward model.
        self.env.core.ppo_reward_model.assert_matches(saved["frozen_reward_state_dict"])
        if self.alg.load(saved, load_cfg, strict):
            self.current_learning_iteration = self.alg.iteration
            self.logger.tot_timesteps = saved["total_timesteps"]
            pool = saved["motion_pool"]
            self.pool_order, self.pool_cursor, self.pool_steps = pool["order"], pool["cursor"], pool["steps"]
            self.training_motion_ids = pool["ids"]
            self.pool_rng.bit_generator.state = pool["rng"]
            self.env.seen_expert_motions.copy_(pool["seen_expert_motions"])
            self.env.core.prepare_training_motion_pool(motion_ids=self.training_motion_ids)
            self.env.reset()
            if self.alg.warming_up:
                self.alg._set_validation_envs(self.env.get_observations()["role"].argmax(-1))
            self._configure_rollout_length()
            torch.set_rng_state(saved["torch_rng"].cpu())
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if "cuda_rng" in saved:
                torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        return saved["infos"]
