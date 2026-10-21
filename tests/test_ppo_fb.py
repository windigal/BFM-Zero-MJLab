"""PPO-FB integration/GAE regressions using real networks and a deterministic CPU VecEnv."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import safetensors.torch
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore
from humanoidverse.agents.envs.utils.gym_spaces import space_to_json
from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluationConfig,
    compute_body_tracking_metrics,
)
from humanoidverse.agents.fb.model import FBModelArchiConfig, FBModelConfig
from humanoidverse.agents.frozen_fb import encode_expert_latents, load_checkpoint_tensors
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig, ForwardArchiConfig
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.ppo_fb.evaluate import evaluate_latents, load_tasks, task_rewards
from humanoidverse.ppo_fb.pretrained import EvaluationAgent, PretrainedController
from humanoidverse.ppo_fb.runner import PpoFbRunner, evaluate_tracking
from humanoidverse.ppo_fb.train import build_parser, build_runner_config
from humanoidverse.ppo_fb.vec_env import ExpertReference, PpoFbVecEnv


def checkpoint(directory):
    space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            for key, dim in {
                "state": 4,
                "privileged_state": 3,
                "last_action": 2,
                "history_actor": 6,
            }.items()
        }
    )
    cfg = FBModelConfig(
        device="cpu",
        seq_length=3,
        actor_std=0.05,
        archi=FBModelArchiConfig(
            z_dim=4,
            b=BackwardArchiConfig(hidden_dim=8, hidden_layers=1, input_filter=DictInputFilterConfig(key=["state", "privileged_state"])),
            f=ForwardArchiConfig(hidden_dim=8, num_parallel=1, input_filter=DictInputFilterConfig(key="state")),
            actor=ActorArchiConfig(
                model="residual",
                hidden_dim=16,
                hidden_layers=2,
                input_filter=DictInputFilterConfig(key=["state", "last_action", "history_actor"]),
            ),
        ),
        obs_normalizer=ObsNormalizerConfig(
            normalizers={key: BatchNormNormalizerConfig() for key in space.keys()},
            allow_mismatching_keys=True,
        ),
    )
    model = cfg.build(space, 2)
    disc_cfg = DiscriminatorArchiConfig(
        hidden_dim=8, hidden_layers=1, input_filter=DictInputFilterConfig(key=["state", "privileged_state"])
    )
    model._discriminator = disc_cfg.build(space, 4)
    root = Path(directory) / "checkpoint" / "model"
    root.mkdir(parents=True)
    config = cfg.model_dump()
    config["archi"]["discriminator"] = disc_cfg.model_dump()
    (root / "config.json").write_text(json.dumps(config))
    (root / "init_kwargs.json").write_text(json.dumps({"obs_space": space_to_json(space), "action_dim": 2}))
    safetensors.torch.save_file(model.state_dict(), root / "model.safetensors")
    return model, root.parent


def expert_buffer():
    episodes = []
    for motion_id, length in enumerate((6, 10)):
        states = torch.arange(length).float()[:, None] / length
        episodes.append(
            {
                "motion_id": torch.full((length,), motion_id),
                "observation": {"state": states.expand(-1, 4).clone(), "privileged_state": states.expand(-1, 3).clone()},
            }
        )
    return TrajectoryDictBuffer(episodes=episodes, seq_length=3)


class PhysicalEnv:
    def __init__(self, num_envs=8):
        self.num_envs, self.device = num_envs, "cpu"
        self.serial = torch.zeros(num_envs)
        self.force_reset = False
        self.base_env = SimpleNamespace(
            num_dof=2,
            config=SimpleNamespace(max_episode_length_s=100),
            dt=1,
            episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
            motion_ids=torch.zeros(num_envs, dtype=torch.long),
            _motion_lib=SimpleNamespace(global_motion_ids=lambda ids: ids, _curr_motion_ids=torch.arange(2)),
            robot_root_states=torch.ones(num_envs, 13),
            tracking_rollout_suspended=False,
            configure_tracking_rollouts=self.configure,
            prepare_training_motion_pool=lambda motion_ids=None: None,
        )

    def configure(self, mask, *, expert_motion_lengths, seq_length):
        self.expert_mask, self.lengths = mask, expert_motion_lengths

    def observation(self):
        state = (self.serial * 10 + self.base_env.episode_length_buf).float()[:, None]
        return {
            "state": state.expand(-1, 4).clone(),
            "privileged_state": state.expand(-1, 3).clone(),
            "last_action": torch.zeros(self.num_envs, 2),
            "history_actor": torch.zeros(self.num_envs, 6),
        }

    def reset(self, to_numpy=False):
        self.base_env.episode_length_buf.zero_()
        return self.observation(), {}

    def step(self, actions, to_numpy=False):
        self.last_action = actions
        core = self.base_env
        core.episode_length_buf += 1
        physical = self.observation()
        representation = core.ppo_reward_model(physical, core.ppo_latent)
        end = self.expert_mask & (core.episode_length_buf >= self.lengths[core.motion_ids] - 1)
        if self.force_reset:
            end[:] = True
        core.episode_length_buf[end] = 0
        self.serial[end] += 1
        return (
            self.observation(),
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            end,
            {
                "tracking_rewards": {"total": self.expert_mask.float()},
                "representation_rewards": representation,
                "aux_rewards": {"penalty_action_rate": actions.square().sum(-1), "limits_dof_pos": torch.zeros(self.num_envs)},
            },
        )

    def close(self):
        pass


class PpoFbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.original, cls.path = checkpoint(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(3)
        self.model = PretrainedController(self.path)
        self.experts = ExpertReference(expert_buffer(), self.model)
        self.physical = PhysicalEnv()
        self.env = PpoFbVecEnv(self.physical, self.model, self.experts, latent_interval=2)
        args = build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--no-wandb",
                "--rollout-steps",
                "3",
                "--epochs",
                "1",
                "--minibatches",
                "2",
                "--eval-interval",
                "0",
                "--warmup-iterations",
                "0",
                "--actor-epochs",
                "1",
                "--warmup-rollout-steps",
                "6",
                "--critic-hidden-dim",
                "16",
            ]
        )
        self.cfg = build_runner_config(args)
        self.runner = PpoFbRunner(self.env, self.cfg)

    def collect(self):
        alg = self.runner.alg
        obs = self.env.get_observations()
        with torch.inference_mode():
            for _ in range(self.cfg["num_steps_per_env"]):
                actions = alg.act(obs)
                obs, rewards, dones, extras = self.env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        return alg

    def test_actor_matches_original_and_normalizer_stays_frozen(self):
        obs = self.env.get_observations()
        expected = self.original.act(dict(obs.items()), obs["latent"])
        actor = self.runner.alg.actor
        torch.testing.assert_close(actor(obs), expected, rtol=0, atol=0)
        norm = copy.deepcopy(actor.obs_normalizer.state_dict())
        actor.train()
        actor.update_normalization(obs)
        self.assertFalse(actor.obs_normalizer.training)
        self.assertFalse(actor.reference_actor.training)
        self.collect().update()
        for key, value in norm.items():
            torch.testing.assert_close(actor.obs_normalizer.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in actor.reference_actor.parameters()))

    def test_prefix_loader_ignores_forward_maps(self):
        state = load_checkpoint_tensors(self.path, ("_actor.",))
        self.assertTrue(state)
        self.assertTrue(all(key.startswith("_actor.") for key in state))

    def test_roles_reference_alignment_and_latent_switch_timeouts(self):
        self.assertEqual(torch.bincount(self.env.roles).tolist(), [4, 2, 2])
        initial = self.env.get_observations()
        saved_latent = initial["latent"].clone()
        self.assertTrue(torch.equal(self.env.get_observations()["history_actor"], initial["history_actor"]))
        for step in (1, 2):
            obs, rewards, done, extras = self.env.step(torch.full((8, 2), 3.0))
            target_z, reference = self.experts.at(torch.zeros(4, dtype=torch.long), torch.full((4,), step), "cpu")
            torch.testing.assert_close(obs["latent"][:4], target_z)
            torch.testing.assert_close(obs["reference"][:4], reference)
            self.assertTrue(torch.all(obs["reference"][4:] == 0))
            self.assertEqual(rewards.shape, (8, 4))
            torch.testing.assert_close(rewards[:, 3], torch.full((8,), -0.06))
            self.assertTrue(torch.all(self.physical.last_action == 1))
            self.assertEqual(done.sum().item(), 4 if step == 2 else 0)
            self.assertTrue(torch.equal(done, extras["time_outs"]))
        torch.testing.assert_close(initial["latent"], saved_latent, rtol=0, atol=0)
        self.assertTrue(torch.all(self.physical.base_env.episode_length_buf == 2))

    def test_reset_reward_uses_previous_latent_and_physical_next_state(self):
        self.physical.force_reset = True
        latent = self.env.get_observations()["latent"].clone()
        before_reset = {"state": torch.ones(8, 4), "privileged_state": torch.ones(8, 3)}
        expected = self.physical.base_env.ppo_reward_model(before_reset, latent)
        obs, rewards, done, _ = self.env.step(torch.zeros(8, 2))
        torch.testing.assert_close(rewards[:, 1:3], expected)
        self.assertTrue(done.all())
        self.assertTrue(torch.all(obs["state"] == 10))
        self.assertTrue(torch.all(self.physical.base_env.episode_length_buf == 0))

    def test_gae_matches_independent_upstream_heads(self):
        alg = self.collect()
        st = alg.storage
        last_values = alg.critic(alg.last_observations).detach()
        for head in range(4):
            original = RolloutStorage("rl", 8, 3, self.env.get_observations(), [2])
            original.values.copy_(st.values[:, :, head : head + 1])
            original.rewards.copy_(st.rewards[:, :, head : head + 1])
            original.dones.copy_(st.dones)
            stub = SimpleNamespace(
                storage=original,
                critic=lambda obs: last_values[:, head : head + 1],
                gamma=alg.gamma,
                lam=alg.lam,
                normalize_advantage_per_mini_batch=False,
            )
            PPO.compute_returns(stub, alg.last_observations)
            torch.testing.assert_close(st.returns[:, :, head : head + 1], original.returns)
            torch.testing.assert_close(st.advantages[:, :, head : head + 1], original.advantages)

    def test_timeout_bootstrap_is_per_head_and_does_not_mutate_reward(self):
        alg = self.runner.alg
        with torch.inference_mode():
            obs = self.env.get_observations()
            alg.act(obs)
            values = alg.transition.values.clone()
            rewards = torch.randn(8, 4)
            timeout = torch.tensor([0, 1] * 4, dtype=torch.bool)
            saved = rewards.clone()
            alg.process_env_step(obs, rewards, timeout, {"time_outs": timeout})
        torch.testing.assert_close(alg.storage.rewards[0], saved + alg.gamma * values * timeout[:, None])
        torch.testing.assert_close(rewards, saved)

    def test_warmup_updates_only_critic_then_actor_and_keeps_critic_rate(self):
        self.cfg["algorithm"].update(
            warmup_iterations=1,
            warmup_patience=2,
            warmup_nrmse=1e3,
            warmup_bias=1e3,
            warmup_motion_coverage=0,
        )
        self.runner = PpoFbRunner(self.env, self.cfg)
        alg = self.runner.alg
        actor = [p.detach().clone() for p in alg.actor_parameters]
        critic = [p.detach().clone() for p in alg.critic.parameters()]
        first = self.collect().update()
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(actor, alg.actor_parameters)))
        self.assertTrue(all(not p.requires_grad for p in alg.actor_parameters))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(critic, alg.critic.parameters())))
        self.assertEqual(first["actor_learning_rate"], 0)
        self.assertTrue(alg.warming_up)
        self.collect().update()
        self.assertFalse(alg.warming_up)
        self.runner._iteration_end(2)
        self.assertEqual(self.cfg["num_steps_per_env"], 3)
        third = self.collect().update()
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(actor, alg.actor_parameters)))
        self.assertEqual(third["critic_learning_rate"], 3e-4)
        self.assertEqual(third["actor_learning_rate"], 3e-5)
        self.assertTrue(all(np.isfinite(value) for value in third.values()))

    def test_disabled_reward_heads_do_not_train_the_critic(self):
        alg = self.runner.alg
        alg.reward_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        before = alg.critic.mlp[-1].weight.detach().clone()
        self.collect().update()
        change = alg.critic.mlp[-1].weight.detach() - before
        self.assertGreater(change[0].abs().sum().item(), 0)
        self.assertEqual(change[1:].abs().sum().item(), 0)

    def start_warmup(self, **options):
        self.cfg["algorithm"].update(warmup_iterations=1, warmup_motion_coverage=0, **options)
        self.runner = PpoFbRunner(self.env, self.cfg)
        return self.collect()

    def test_unclipped_critic_escapes_the_value_clipping_plateau(self):
        alg = self.collect()
        self.assertFalse(alg.use_clipped_value_loss)
        with torch.no_grad():
            alg.critic.mlp[-1].weight.zero_()
            alg.critic.mlp[-1].bias.fill_(1)
            alg.storage.values.zero_()
            alg.storage.returns.fill_(42)
        before = alg.critic.mlp[-1].bias.detach().clone()
        alg._update_critic(False)
        # At V_old=0, V=1, return=42, the clipped max loss has zero gradient.
        self.assertTrue(torch.all(alg.critic.mlp[-1].bias > before))

    def test_validation_detects_constant_bias_even_with_perfect_explained_variance(self):
        alg = self.start_warmup()
        with torch.no_grad():
            for parameter in alg.critic.parameters():
                parameter.zero_()
            alg.storage.returns.fill_(42)
        metrics = alg._validate_critic()
        self.assertEqual(metrics["validation/expert/track/explained_variance"], 1)
        self.assertEqual(metrics["validation/expert/track/bias"], 1)
        self.assertEqual(alg.validation_streak, 0)
        self.assertTrue(alg.warming_up)

    def test_warmup_returns_use_long_lambda_one_targets(self):
        alg = self.start_warmup()
        with torch.no_grad():
            for parameter in alg.critic.parameters():
                parameter.zero_()
            alg.storage.rewards.fill_(1)
            alg.storage.values.zero_()
            alg.storage.dones.zero_()
            alg.compute_returns(alg.last_observations)
        remaining = torch.arange(alg.storage.num_transitions_per_env, 0, -1)
        expected = (1 - alg.gamma**remaining) / (1 - alg.gamma)
        torch.testing.assert_close(alg.storage.returns[:, 0, 0], expected)

    def test_unfreeze_rechecks_validation_after_the_last_critic_fit(self):
        alg = self.start_warmup(warmup_patience=1)
        with torch.no_grad():
            for parameter in alg.critic.parameters():
                parameter.zero_()
            alg.storage.returns.zero_()

        def unstable_fit(warmup):
            with torch.no_grad():
                alg.critic.mlp[-1].bias.fill_(1000)
            return torch.zeros(4), 1

        with patch.object(alg, "_update_critic", side_effect=unstable_fit):
            alg.update()
        self.assertTrue(alg.warming_up)
        self.assertEqual(alg.validation_streak, 0)

    def test_validation_uses_untrained_trajectories_and_runs_before_fitting(self):
        alg = self.start_warmup()
        seen_roles = set()
        for batch in alg.storage.mini_batch_generator(2, 1, ~alg.validation_envs):
            seen_roles.update(batch.observations["role"].argmax(-1).tolist())
        self.assertEqual(seen_roles, {0, 1, 2})
        validation = alg.storage.observations[:, alg.validation_envs].flatten(0, 1)
        # Mark held-out trajectories with inputs that cannot occur in training.
        alg.storage.observations["state"][:, alg.validation_envs] = 999
        for batch in alg.storage.mini_batch_generator(2, 1, ~alg.validation_envs):
            self.assertFalse((batch.observations["state"] == 999).any())
        self.assertEqual(len(validation), alg.storage.num_transitions_per_env * 3)
        with patch.object(alg, "_validate_critic", wraps=alg._validate_critic) as validate:
            original = alg._update_critic

            def fit(warmup):
                self.assertEqual(validate.call_count, 1)
                return original(warmup)

            with patch.object(alg, "_update_critic", side_effect=fit):
                alg.update()

    def test_warmup_coverage_blocks_unfreezing_and_timeout_preserves_frozen_checkpoint(self):
        self.cfg["algorithm"].update(
            warmup_iterations=1,
            warmup_max_iterations=1,
            warmup_nrmse=1e3,
            warmup_bias=1e3,
            warmup_patience=1,
            warmup_motion_coverage=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            self.runner = PpoFbRunner(self.env, self.cfg, directory)
            with self.assertRaisesRegex(RuntimeError, "warmup did not converge"):
                self.runner.learn(2)
            self.assertTrue(self.runner.alg.warming_up)
            self.assertEqual(self.env.expert_motion_coverage(), 0.5)
            saved = torch.load(Path(directory) / "model_1.pt", weights_only=False)
            self.assertTrue(saved["warming_up"])
            self.assertEqual(saved["validation_streak"], 0)
            self.env.seen_expert_motions.zero_()
            self.runner.alg.warming_up = False
            self.runner.alg._set_actor_trainable()
            self.runner.alg.validation_envs.zero_()
            self.runner.load(Path(directory) / "model_1.pt")
            self.assertTrue(self.runner.alg.warming_up)
            self.assertTrue(all(not parameter.requires_grad for parameter in self.runner.alg.actor_parameters))
            self.assertEqual(self.env.expert_motion_coverage(), 0.5)
            self.assertEqual(self.runner.alg.validation_envs.sum().item(), 3)
            self.assertEqual(self.runner.alg.storage.num_transitions_per_env, 6)

    def test_kl_stop_leaves_actor_fixed_while_critic_keeps_updating(self):
        alg = self.collect()
        actor = [p.detach().clone() for p in alg.actor_parameters]
        critic = [p.detach().clone() for p in alg.critic.parameters()]
        alg.actor_epochs = 3
        with patch.object(alg.actor, "get_kl_divergence", return_value=torch.ones(12)):
            metrics = alg.update()
        self.assertEqual(metrics["actor_updates"], 0)
        self.assertEqual(metrics["kl_stopped"], 1)
        self.assertEqual(metrics["critic_updates"], 2)
        self.assertEqual(metrics["actor_learning_rate"], 3e-5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(actor, alg.actor_parameters)))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(critic, alg.critic.parameters())))

    def test_actor_and_critic_have_independent_update_budgets(self):
        alg = self.collect()
        alg.actor_epochs, alg.num_learning_epochs = 1, 3
        alg.desired_kl = 10
        metrics = alg.update()
        self.assertEqual(metrics["actor_updates"], 2)
        self.assertEqual(metrics["critic_updates"], 6)
        self.assertEqual(metrics["actor_learning_rate"], 3e-5)

    def test_motion_pool_refresh_runs_without_evaluation_and_restores_live_observation(self):
        alg = self.collect()
        self.cfg["motion_pool_refresh_steps"] = 3
        self.cfg["eval_interval"] = 0
        live = alg.last_observations
        with patch.object(self.env.core, "prepare_training_motion_pool") as reload_pool:
            self.runner._iteration_end(1)
        reload_pool.assert_called_once_with(motion_ids=self.runner.training_motion_ids)
        self.assertTrue(torch.all(live["state"] == 0))
        self.assertEqual(self.runner.pool_steps, 0)
        self.assertEqual(sorted(self.runner.training_motion_ids), [0, 1])

    def test_pool_permutation_covers_all_ids_before_repeating(self):
        self.cfg["motion_pool_size"] = 8
        self.runner.pool_order = np.arange(19)
        self.runner.pool_cursor = 0
        visited = []
        for _ in range(3):
            self.runner._advance_motion_pool()
            visited.extend(self.runner.training_motion_ids)
        self.assertEqual(visited, list(range(19)))

    def test_motion_pool_size_is_independent_of_environment_count(self):
        self.cfg["motion_pool_size"] = 3
        self.runner.pool_order = np.arange(19)
        self.runner.pool_cursor = 0
        self.runner._advance_motion_pool()
        self.assertEqual(self.runner.training_motion_ids, [0, 1, 2])

    def test_full_tracking_evaluation_limits_loaded_pool_size(self):
        chunks = []
        library = SimpleNamespace(_num_unique_motions=5000, load_motions=lambda **kwargs: chunks.append(kwargs["motion_ids"]))
        env = SimpleNamespace(_env=SimpleNamespace(_motion_lib=library, num_envs=4096))
        evaluator = HumanoidVerseMjlabTrackingEvaluationConfig(motion_batch_size=1024).build()
        with patch("humanoidverse.agents.evaluations.humanoidverse_mjlab._async_tracking_worker", return_value={}):
            evaluator.run(timestep=0, agent_or_model=None, logger=None, env=env)
        self.assertEqual([len(chunk) for chunk in chunks], [1024, 1024, 1024, 1024, 904])
        self.assertEqual([motion_id for chunk in chunks for motion_id in chunk], list(range(5000)))

    def test_step_budget_counts_long_warmup_rollouts(self):
        self.cfg["algorithm"].update(warmup_iterations=50)
        self.cfg["max_env_steps"] = 48
        self.runner = PpoFbRunner(self.env, self.cfg)
        self.runner.learn(20)
        self.assertEqual(self.runner.alg.iteration, 1)
        self.assertEqual(self.runner.logger.tot_timesteps, 48)
        self.assertTrue(self.runner.alg.warming_up)

    def test_tracking_subset_is_seeded_and_not_the_dataset_prefix(self):
        library = SimpleNamespace(_num_unique_motions=5000, load_motions=lambda **kwargs: None)
        env = SimpleNamespace(_env=SimpleNamespace(_motion_lib=library, num_envs=64))

        def selection(seed):
            evaluator = HumanoidVerseMjlabTrackingEvaluationConfig(max_motions=64, selection_seed=seed).build()
            with patch("humanoidverse.agents.evaluations.humanoidverse_mjlab._async_tracking_worker", return_value={}):
                evaluator.run(timestep=0, agent_or_model=None, logger=None, env=env)
            return evaluator.motion_ids

        ids = selection(42)
        self.assertEqual(ids, selection(42))
        self.assertNotEqual(ids, selection(43))
        self.assertEqual(len(set(ids)), 64)
        self.assertNotEqual(ids, list(range(64)))
        self.assertGreater(max(ids), 4500)

    def test_runner_loop_checkpoint_roundtrip(self):
        self.runner.learn(2)
        self.assertEqual(self.runner.alg.iteration, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            self.runner.save(path)
            self.runner.alg.learning_rate = 9
            parameter = next(self.model._backward_map.parameters())
            original = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(1)
            with self.assertRaisesRegex(ValueError, "frozen rewards differ"):
                self.runner.load(path)
            with torch.no_grad():
                parameter.copy_(original)
            self.runner.load(path)
            self.assertEqual(self.runner.current_learning_iteration, 2)
            self.assertLess(self.runner.alg.learning_rate, 1e-3)
            self.runner.learn(1)
            self.assertEqual(self.runner.alg.iteration, 3)

    def test_wandb_checkpoint_stays_local(self):
        self.runner.logger.logger_type = "wandb"
        self.runner.logger.writer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.runner.save(str(path))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["iter"], self.runner.alg.iteration)
            self.assertIn("actor_state_dict", saved)
            self.assertIn("optimizer_state_dict", saved)
        self.runner.logger.writer.save_model.assert_not_called()

    def test_eval_hook_refreshes_the_tensor_held_by_base_loop(self):
        alg = self.collect()
        obs = alg.last_observations
        self.runner.cfg["eval_interval"] = 1

        def evaluate(*args, **kwargs):
            self.physical.serial += 10
            return {}

        with patch("humanoidverse.ppo_fb.runner.evaluate_tracking", side_effect=evaluate):
            self.runner._iteration_end(1)
        self.assertTrue(torch.all(obs["state"] == 100))
        self.assertTrue(alg.actor.training)

    def test_single_head_gae_and_switch_ablation(self):
        self.env.single_head = True
        self.env.truncate_latent_switch = False
        self.runner = PpoFbRunner(self.env, self.cfg)
        alg = self.collect()
        self.assertEqual(alg.storage.returns.shape, (3, 8, 1))
        self.assertFalse(alg.storage.dones.any())
        original = RolloutStorage("rl", 8, 3, self.env.get_observations(), [2])
        for name in ("values", "rewards", "dones"):
            getattr(original, name).copy_(getattr(alg.storage, name))
        stub = SimpleNamespace(storage=original, critic=alg.critic, gamma=alg.gamma, lam=alg.lam, normalize_advantage_per_mini_batch=False)
        with torch.no_grad():
            PPO.compute_returns(stub, alg.last_observations)
        torch.testing.assert_close(alg.storage.returns, original.returns)
        torch.testing.assert_close(alg.storage.advantages, original.advantages)
        self.assertTrue(all(np.isfinite(v) for v in alg.update().values()))

    def test_bf16_expert_windows_accumulate_in_float32(self):
        observations = {"state": torch.randn(500, 4), "privileged_state": torch.randn(500, 3)}
        with patch.object(self.model, "backward_map", side_effect=lambda obs: self.original.backward_map(obs).bfloat16()):
            actual = encode_expert_latents(self.model, observations, seq_length=8)
            embedding = self.model.backward_map(observations).float()
        expected = self.model.project_z(torch.stack([embedding[i : i + 8].mean(0) for i in range(500)]))
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

    def test_cross_returns_and_eval_reward_cleanup(self):
        latents = self.model.sample_z(2)
        previous = self.physical.base_env.ppo_reward_model
        report = evaluate_latents(self.physical, EvaluationAgent(self.model), latents, horizon=3)
        matrix = torch.tensor(report["cross_returns"])
        self.assertEqual(matrix.shape, (2, 2))
        # Both conditional policies visit the same physical states in this deterministic env.
        torch.testing.assert_close(matrix[:, 0], matrix[:, 1])
        self.assertAlmostEqual(report["mean_fb"], matrix.diag().mean().item())
        self.assertEqual(report["fall_rate"], 0)
        self.assertIs(self.physical.base_env.ppo_reward_model, previous)

    def test_reward_tasks_are_held_out_and_have_known_rewards(self):
        tasks = [{"name": "test", "observation": "state", "indices": [0], "target": [0], "scale": 2}]
        rewards = task_rewards({"state": torch.tensor([[0.0], [2.0]])}, tasks)
        torch.testing.assert_close(rewards[:, 0], torch.tensor([1.0, np.exp(-1)], dtype=torch.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps({"train": [], "eval": tasks}))
            self.assertEqual(load_tasks(path), tasks)
            path.write_text(json.dumps({"train": tasks, "eval": tasks}))
            with self.assertRaises(ValueError):
                load_tasks(path)

    def test_eval_failure_also_restores_training_observations(self):
        alg = self.collect()
        self.runner.cfg["eval_interval"] = 1
        with patch("humanoidverse.ppo_fb.runner.evaluate_tracking", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.runner._iteration_end(1)
        self.assertTrue(torch.all(alg.last_observations["state"] == 0))
        self.assertTrue(alg.actor.training)

    def test_tracking_eval_uses_inference_mode_and_returns_aggregate(self):
        def run(**kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            return {"motion": {"emd": 1.0}}, {"emd": 1.0}

        def build(config):
            self.assertEqual(config.seq_length, 1)
            self.assertEqual(config.motion_batch_size, 1024)
            return SimpleNamespace(run=run)

        with patch("humanoidverse.ppo_fb.runner.HumanoidVerseMjlabTrackingEvaluationConfig.build", autospec=True, side_effect=build):
            self.assertEqual(evaluate_tracking(self.physical, self.model), {"emd": 1.0})

    def test_body_mpjpe_uses_positions_and_separates_translation(self):
        target = torch.zeros(3, 2, 3)
        metrics = compute_body_tracking_metrics(target + 0.1, target)
        self.assertAlmostEqual(metrics["mpjpe_body_global_mm"], 100 * 3**0.5, places=4)
        self.assertEqual(metrics["mpjpe_body_local_mm"], 0)
        self.assertEqual(metrics["success_body_0.5m"], 1)
        current = target.clone()
        current[:, 1, 0] = 1
        metrics = compute_body_tracking_metrics(current, target)
        self.assertEqual(metrics["mpjpe_body_local_mm"], 500)
        self.assertEqual(metrics["success_body_0.5m"], 0)


class CoreRewardOrderTest(unittest.TestCase):
    def test_reward_observation_is_scaled_cleanly_with_an_explicit_noise_ablation(self):
        core = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        core._compute_reference_and_privileged_obs = lambda: None
        core.default_dof_pos = core.default_dof_pos_offset = torch.zeros(1, 2)
        core.dof_pos = core.dof_vel = torch.ones(1, 2)
        core.projected_gravity = core.base_ang_vel = torch.ones(1, 3)
        core._max_local_self = torch.ones(1, 4)
        core.config = SimpleNamespace(obs=SimpleNamespace(obs_scales={"base_ang_vel": 0.25, "dof_vel": 2.0}))
        core.ppo_reward_observation_noise = False
        with patch.object(core, "_apply_obs_scale_noise", side_effect=lambda key, value: value * 3) as noise:
            observation = core._reward_observation()
            noise.assert_not_called()
            torch.testing.assert_close(observation["state"], torch.tensor([[1, 1, 2, 2, 1, 1, 1, 0.25, 0.25, 0.25]]))
            torch.testing.assert_close(observation["privileged_state"], torch.ones(1, 4))
            core.ppo_reward_observation_noise = True
            noisy = core._reward_observation()
            self.assertEqual(noise.call_count, 5)
            torch.testing.assert_close(noisy["state"], torch.full((1, 10), 3.0))

    def test_core_evaluates_representation_before_same_step_reset(self):
        core = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        core.device = "cpu"
        core.actions = core.last_actions = core.last_dof_vel = core.dof_vel = torch.zeros(1, 2)
        core._normalized_action = lambda x: x
        core._mjlab_action_input = lambda: core.actions
        core.mjlab_env = SimpleNamespace(step=lambda actions: (None, None, torch.tensor([False]), torch.tensor([True]), {}))
        core._refresh_state = lambda: setattr(core, "physical_state", 9.0)
        core._compute_reward = lambda: (torch.ones(1), {})
        core._reward_observation = lambda: {"state": torch.tensor([[core.physical_state]])}
        core.ppo_reward_model = lambda obs, z: obs["state"].clone()
        core.ppo_latent = torch.ones(1, 4)
        core.tracking_rollout_enabled = core.tracking_rollout_suspended = core.is_evaluating = False
        core.tracking_rollout_suspended = True
        core.use_reward_penalty_curriculum = core.add_noise_currculum = False
        core.extras = {}
        core.episode_length_buf = torch.ones(1, dtype=torch.long)
        core.last_episode_length_buf = torch.zeros(1, dtype=torch.long)
        core._update_average_episode_length = lambda ids: None
        core._update_reward_penalty_curriculum = core._update_obs_noise_curriculum = lambda: None
        core.reset_idx = lambda ids: setattr(core, "physical_state", -1.0)
        _, _, _, info = core.step(torch.zeros(1, 2))
        self.assertEqual(info["representation_rewards"].item(), 9.0)
        self.assertEqual(core.physical_state, -1.0)


if __name__ == "__main__":
    unittest.main()
