"""Tests for the pieces the PPO-FB branch relies on.

These need only torch (no mjlab / GPU): the motion lib is built through
``__new__`` and the buffers are driven with toy rollouts, so they run anywhere.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import gymnasium
import numpy as np
import safetensors.torch
import torch

from humanoidverse.agents import frozen_fb
from humanoidverse.agents.buffers import trajectory as traj
from humanoidverse.agents.envs.humanoidverse_mjlab import TRACKING_BODIES, HumanoidVerseMjlabCore
from humanoidverse.agents.fb.model import FBModelArchiConfig, FBModelConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import BackwardArchiConfig, ForwardArchiConfig, SimpleActorArchiConfig
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.torch_utils import _sqrt_positive_part, matrix_to_quaternion, quaternion_to_matrix


class RotationConversionTest(unittest.TestCase):
    def test_positive_sqrt_preserves_values_and_zero_subgradient(self):
        x = torch.tensor([-4.0, -0.0, 0.0, 1e-12, 0.25, 4.0], dtype=torch.float64, requires_grad=True)
        y = _sqrt_positive_part(x)
        torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 0.0, 1e-6, 0.5, 2.0], dtype=x.dtype))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.0, 0.0, 5e5, 1.0, 0.25], dtype=x.dtype))

    def test_matrix_quaternion_round_trip_for_motion_shaped_and_single_inputs(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(1, 300, 31, 4, generator=generator, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        q[0, :4, 0] = torch.eye(4, dtype=q.dtype)
        matrices = quaternion_to_matrix(q)
        for matrix in (matrices, matrices.transpose(1, 2), matrices[0, 0, 0]):
            actual = matrix_to_quaternion(matrix)
            torch.testing.assert_close(quaternion_to_matrix(actual), matrix)
            torch.testing.assert_close(actual.norm(dim=-1), torch.ones_like(actual[..., 0]))


def _expert_buffer(lengths: list[int], seq_length: int = 8) -> traj.TrajectoryDictBuffer:
    episodes = [
        {
            "observation": {"x": (torch.arange(n, dtype=torch.float32) + 1000 * i)[:, None]},
            "motion_id": torch.full((n,), i),
        }
        for i, n in enumerate(lengths)
    ]
    buffer = traj.TrajectoryDictBuffer(episodes=episodes, seq_length=seq_length, device="cpu")
    buffer._get_idxs = traj.get_idxs  # skip torch.compile
    return buffer


def _replay_with_resets(*, n_iters, capacity, episode_len, eval_every, num_envs=3):
    """Fill a replay buffer using the same storage rules as the training loop.

    The env resets inside ``step()`` and returns the post-reset observation, so a
    row's ``truncated`` flag is whether the step *taken from that row* ended the
    episode (``new_done``).
    """
    buffer = traj.TrajectoryDictBufferMultiDim(
        capacity=capacity,
        device="cpu",
        n_dim=2,
        end_key="truncated",
        output_key_t=["observation"],
        output_key_tp1=["observation", "terminated"],
    )
    buffer._get_idxs = traj.get_idxs
    episode = np.arange(num_envs)
    step_in_episode = np.array([0, 2, 4])[:num_envs]
    next_episode = num_envs
    terminated = np.zeros(num_envs, dtype=bool)
    for it in range(n_iters):
        if it > 0 and it % eval_every == 0:  # evaluation fully resets every env
            for env in range(num_envs):
                episode[env] = next_episode
                next_episode += 1
            step_in_episode[:] = 0
        obs = (episode * 1000 + step_in_episode).astype(np.float32)
        step_next = step_in_episode + 1
        new_truncated = step_next >= episode_len
        if (it + 1) % eval_every == 0:  # next iteration evaluates -> forced truncation
            new_truncated = np.ones(num_envs, dtype=bool)
        new_done = np.logical_or(np.zeros(num_envs, dtype=bool), new_truncated)
        buffer.extend(
            {
                "observation": {"x": torch.tensor(obs)[None, :, None]},
                "terminated": torch.tensor(terminated)[None, :, None],
                "truncated": torch.tensor(new_done)[None, :, None],
            }
        )
        for env in range(num_envs):
            if step_next[env] >= episode_len and not ((it + 1) % eval_every == 0):
                episode[env] = next_episode
                next_episode += 1
                step_in_episode[env] = 0
            else:
                step_in_episode[env] = step_next[env]
    return buffer


class ExpertBufferHelpersTest(unittest.TestCase):
    def test_motion_length_by_global_id(self) -> None:
        buffer = _expert_buffer([12, 9, 20])
        self.assertEqual([buffer.motion_length(i) for i in range(3)], [12, 9, 20])
        with self.assertRaises(KeyError):
            buffer.motion_length(99)

    def test_observation_sequence_is_exact_and_bounds_checked(self) -> None:
        buffer = _expert_buffer([12, 9, 20])
        window = buffer.observation_sequence(2, 3, 5, next_offset=1)["x"].flatten().tolist()
        self.assertEqual(window, [2004.0 + k for k in range(5)])
        with self.assertRaises(IndexError):
            buffer.observation_sequence(0, 5, 8, next_offset=1)

    def test_observations_at_maps_global_ids_and_rejects_bad_input(self) -> None:
        buffer = _expert_buffer([12, 9, 20])
        got = buffer.observations_at(torch.tensor([0, 1, 2]), torch.tensor([0, 8, 19]))["x"].flatten().tolist()
        self.assertEqual(got, [0.0, 1008.0, 2019.0])
        with self.assertRaises(IndexError):
            buffer.observations_at(torch.tensor([1]), torch.tensor([9]))
        with self.assertRaises(KeyError):
            buffer.observations_at(torch.tensor([7]), torch.tensor([0]))

    def test_expert_windows_never_cross_motions_and_reach_last_frame(self) -> None:
        buffer = _expert_buffer([12, 9, 20])
        torch.manual_seed(0)
        crossed, reached = 0, set()
        for _ in range(500):
            batch = buffer.sample(64, seq_length=8)
            x = batch["observation"]["x"].view(-1, 8)
            nxt = batch["next"]["observation"]["x"].view(-1, 8)
            crossed += int(((x // 1000) != (nxt // 1000)).any(dim=1).sum())
            crossed += int(((x[:, 1:] - x[:, :-1]) != 1).any(dim=1).sum())
            reached |= {int(v) for v in nxt[:, -1].tolist()}
        self.assertEqual(crossed, 0)
        self.assertEqual(max(v for v in reached if v // 1000 == 1) - 1000, 8)  # last frame of a 9-step motion


class ReplayResetLayoutTest(unittest.TestCase):
    """The same-step-reset storage layout must never produce a cross-episode pair."""

    def _check(self, buffer) -> tuple[set, int]:
        torch.manual_seed(0)
        pairs = set()
        for _ in range(1500):
            batch = buffer.sample(8)
            for a, b in zip(
                batch["observation"]["x"].flatten().tolist(),
                batch["next"]["observation"]["x"].flatten().tolist(),
            ):
                pairs.add((int(a), int(b)))
        invalid = {p for p in pairs if p[0] // 1000 != p[1] // 1000 or p[1] != p[0] + 1}
        return invalid, len(pairs)

    def test_no_invalid_transitions_including_ring_wraparound(self) -> None:
        for n_iters in (30, 41, 47, 55, 61, 73):
            with self.subTest(n_iters=n_iters):
                buffer = _replay_with_resets(n_iters=n_iters, capacity=40, episode_len=6, eval_every=17)
                invalid, total = self._check(buffer)
                self.assertEqual(invalid, set())
                self.assertGreater(total, 0)

    def test_first_transition_of_each_episode_is_sampleable(self) -> None:
        buffer = _replay_with_resets(n_iters=61, capacity=40, episode_len=6, eval_every=17)
        torch.manual_seed(0)
        starts = set()
        for _ in range(1500):
            batch = buffer.sample(8)
            starts |= {int(v) for v in batch["observation"]["x"].flatten().tolist() if int(v) % 1000 == 0}
        self.assertTrue(starts, "no episode-initial observation was ever sampled as a transition start")

    def test_sampler_reaches_every_legal_start_and_never_the_last_row(self) -> None:
        """Two complete 6-step episodes, no wrap, no eval: legal starts are exactly k=0..4.

        Row k=5 is the last row of its episode, so its "next" row belongs to the following
        episode and must never be a start. Row k=4 (the final real transition 4 -> 5) used to
        be unreachable with ``end_point = L - seq - 1``.
        """
        buffer = traj.TrajectoryDictBufferMultiDim(
            capacity=64,
            device="cpu",
            n_dim=2,
            end_key="truncated",
            output_key_t=["observation"],
            output_key_tp1=["observation"],
        )
        buffer._get_idxs = traj.get_idxs
        episode_len = 6
        for episode in range(2):
            for k in range(episode_len):
                buffer.extend(
                    {
                        "observation": {"x": torch.tensor([[[float(episode * 1000 + k)]]])},
                        "truncated": torch.tensor([[[k == episode_len - 1]]]),
                    }
                )
        torch.manual_seed(0)
        starts = set()
        for _ in range(400):
            starts |= {int(v) % 1000 for v in buffer.sample(8)["observation"]["x"].flatten().tolist()}
        self.assertEqual(starts, {0, 1, 2, 3, 4})

    def test_get_full_buffer_has_no_invalid_pairs(self) -> None:
        buffer = _replay_with_resets(n_iters=61, capacity=40, episode_len=6, eval_every=17)
        full = buffer.get_full_buffer()
        a = full["observation"]["x"].flatten().tolist()
        b = full["next"]["observation"]["x"].flatten().tolist()
        self.assertTrue(all(int(x) // 1000 == int(y) // 1000 for x, y in zip(a, b)))


class MotionLibGlobalIdTest(unittest.TestCase):
    def _motion_lib(self, num_motions: int = 6) -> MotionLibBase:
        lib = MotionLibBase.__new__(MotionLibBase)
        lib._device = "cpu"
        lib._num_unique_motions = num_motions
        lib._motion_source_names = ["complete_dataset"]
        lib._motion_source_weights = torch.ones(1)
        lib._motion_source_ids = torch.zeros(num_motions, dtype=torch.long)
        lib._sampling_prob = torch.full((num_motions,), 1.0 / num_motions)
        lib._curr_motion_ids = None
        lib._global_emd_sampling = False
        lib.all_motions_loaded = True
        return lib

    def test_global_motion_ids_maps_pool_to_dataset_ids(self) -> None:
        lib = self._motion_lib()
        with self.assertRaises(RuntimeError):
            lib.global_motion_ids(torch.tensor([0]))
        lib._curr_motion_ids = torch.tensor([4, 1, 5])
        self.assertEqual(lib.global_motion_ids(torch.tensor([2, 0, 1])).tolist(), [5, 4, 1])

    def test_set_global_emd_sampling_validates_and_normalizes(self) -> None:
        lib = self._motion_lib()
        lib.set_global_emd_sampling(torch.tensor([1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))
        self.assertAlmostEqual(float(lib._sampling_prob.sum()), 1.0, places=6)
        self.assertTrue(lib._global_emd_sampling)
        self.assertFalse(lib.all_motions_loaded)
        for bad in (torch.ones(5), torch.zeros(6), torch.tensor([1.0, -1.0, 1, 1, 1, 1]), torch.full((6,), float("nan"))):
            with self.assertRaises(ValueError):
                lib.set_global_emd_sampling(bad)

    def test_partial_pool_is_not_reweighted_twice(self) -> None:
        lib = self._motion_lib()
        lib.set_global_emd_sampling(torch.tensor([0.5, 0.25, 0.1, 0.05, 0.05, 0.05]))
        lib._curr_motion_ids = torch.tensor([0, 0, 1, 2])  # partial importance-sampled pool
        lib._refresh_sampling_batch_prob()
        self.assertTrue(torch.allclose(lib._sampling_batch_prob, torch.full((4,), 0.25)))
        # a fully resident pool keeps the direct weights
        lib._curr_motion_ids = torch.arange(6)
        lib._refresh_sampling_batch_prob()
        self.assertTrue(torch.allclose(lib._sampling_batch_prob, lib._sampling_prob))


def _tracking_env(num_envs: int = 3, num_bodies: int = len(TRACKING_BODIES)) -> HumanoidVerseMjlabCore:
    """A bare core env holding only what the tracking reward reads."""
    env = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
    env.num_envs, env.device = num_envs, "cpu"
    env.tracking_rollout_enabled, env.tracking_rollout_suspended, env.is_evaluating = True, False, False
    env.tracking_expert_mask = torch.tensor([True, False, True])[:num_envs]
    env.tracking_body_indices = torch.arange(num_bodies)
    gen = torch.Generator().manual_seed(0)
    pos = torch.randn(num_envs, num_bodies, 3, generator=gen) * 0.3
    rot = torch.zeros(num_envs, num_bodies, 4)
    rot[..., 3] = 1.0
    vel = torch.randn(num_envs, num_bodies, 3, generator=gen) * 0.1
    ang = torch.randn(num_envs, num_bodies, 3, generator=gen) * 0.1
    env.ref_body_pos_extend, env.ref_body_rot_extend = pos, rot
    env.ref_body_vel_extend, env.ref_body_ang_vel_extend = vel, ang
    env._current = [pos.clone(), rot.clone(), vel.clone(), ang.clone()]
    env._extend_body_state = lambda: tuple(x.clone() for x in env._current)
    return env


class TrackingRewardTest(unittest.TestCase):
    def test_perfect_tracking_scores_one_only_on_masked_envs(self) -> None:
        rewards = _tracking_env()._compute_tracking_rewards()
        for name, value in rewards.items():
            expected = [1.0, 0.0, 1.0]
            self.assertTrue(torch.allclose(value, torch.tensor(expected), atol=1e-5), name)

    def test_xy_translation_is_ignored_but_height_error_is_penalized(self) -> None:
        for delta, expected in (([0.3, 0, 0], 1.0), ([0, 0.3, 0], 1.0), ([0, 0, 0.3], math.exp(-1.0))):
            with self.subTest(delta=delta):
                env = _tracking_env()
                env._current[0] = env._current[0] + torch.tensor(delta)
                got = float(env._compute_tracking_rewards()["relative_body_position"][0])
                self.assertAlmostEqual(got, expected, places=4)

    def test_rigid_yaw_preserves_relative_pose_and_penalizes_global_motion(self) -> None:
        env = _tracking_env()
        yaw = 0.5
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        quat = torch.zeros(len(TRACKING_BODIES), 4)
        quat[:, 2], quat[:, 3] = math.sin(yaw / 2), math.cos(yaw / 2)
        anchor = TRACKING_BODIES.index("torso_link")
        for idx in (0, 2):
            ref = env.ref_body_pos_extend[idx]
            env._current[0][idx] = (ref - ref[anchor]) @ rotation.T + ref[anchor]
            env._current[1][idx] = quat
            env._current[2][idx] = env.ref_body_vel_extend[idx] @ rotation.T
            env._current[3][idx] = env.ref_body_ang_vel_extend[idx] @ rotation.T
        rewards = env._compute_tracking_rewards()
        self.assertAlmostEqual(float(rewards["relative_body_position"][0]), 1.0, places=4)
        self.assertAlmostEqual(float(rewards["relative_body_orientation"][0]), 1.0, places=4)
        self.assertAlmostEqual(float(rewards["anchor_orientation"][0]), math.exp(-(yaw**2) / 0.4**2), places=4)
        self.assertLess(float(rewards["global_body_linear_velocity"][0]), 1.0)
        self.assertLess(float(rewards["global_body_angular_velocity"][0]), 1.0)

    def test_suspended_or_evaluating_rollouts_give_zero_reward(self) -> None:
        for attr in ("tracking_rollout_suspended", "is_evaluating"):
            with self.subTest(attr=attr):
                env = _tracking_env()
                setattr(env, attr, True)
                self.assertEqual(float(env._compute_tracking_rewards()["total"].abs().sum()), 0.0)

    def test_tracking_bodies_exist_in_the_g1_model(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path("humanoidverse/data/robots/g1_mjlab/g1_29dof.xml")
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
        self.assertEqual([name for name in TRACKING_BODIES if name not in names], [])


def _tiny_fb_model(*, state_dim: int = 5, z_dim: int = 8, seed: int = 0, prepare_for_train: bool = False, seq_length: int = 1):
    """A small FBModel built from the real config classes, mirroring the production layout."""
    torch.manual_seed(seed)
    obs_space = gymnasium.spaces.Dict(
        {
            "state": gymnasium.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32),
            "privileged_state": gymnasium.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "last_action": gymnasium.spaces.Box(-1, 1, (3,), np.float32),
        }
    )
    keys = ["state", "privileged_state", "last_action"]
    cfg = FBModelConfig(
        device="cpu",
        seq_length=seq_length,
        archi=FBModelArchiConfig(
            z_dim=z_dim,
            f=ForwardArchiConfig(hidden_dim=16, hidden_layers=1, embedding_layers=2, input_filter=DictInputFilterConfig(key=keys)),
            b=BackwardArchiConfig(hidden_dim=16, hidden_layers=1, input_filter=DictInputFilterConfig(key=["state", "privileged_state"])),
            actor=SimpleActorArchiConfig(hidden_dim=16, hidden_layers=1, embedding_layers=2, input_filter=DictInputFilterConfig(key=keys)),
        ),
        obs_normalizer=ObsNormalizerConfig(
            normalizers={k: BatchNormNormalizerConfig(momentum=0.01) for k in keys}, allow_mismatching_keys=True
        ),
    )
    model = cfg.build(obs_space, 3)
    if prepare_for_train:
        model._prepare_for_train()
    return model


def _save_pretrained(model, root: Path, *, nested: bool = False) -> Path:
    target = root / "checkpoint" if nested else root
    (target / "model").mkdir(parents=True)
    safetensors.torch.save_model(model, target / "model" / "model.safetensors")
    (target / "model" / "config.json").write_text(json.dumps(model.cfg.model_dump()))
    return root


def _perturb(model) -> None:
    """Give B and the normalizer distinctive values so a copy is detectable."""
    with torch.no_grad():
        for p in model._backward_map.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        for name, buf in model._obs_normalizer.named_buffers():
            if buf.dtype.is_floating_point:
                buf.add_(torch.rand_like(buf) + 1.0)


class FrozenRepresentationTest(unittest.TestCase):
    def test_resolve_accepts_run_dir_and_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(_tiny_fb_model(), Path(tmp) / "run", nested=True)
            self.assertEqual(frozen_fb.resolve_fb_checkpoint(run), (run / "checkpoint").resolve())
            self.assertEqual(frozen_fb.resolve_fb_checkpoint(run / "checkpoint"), (run / "checkpoint").resolve())
            self.assertEqual(frozen_fb.pretrained_z_dim(run), 8)
            with self.assertRaises(FileNotFoundError):
                frozen_fb.resolve_fb_checkpoint(Path(tmp) / "missing")

    def test_loads_only_b_and_normalizer_and_leaves_f_and_actor_alone(self) -> None:
        pretrained = _tiny_fb_model(seed=1)
        _perturb(pretrained)
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(pretrained, Path(tmp) / "run", nested=True)
            fresh = _tiny_fb_model(seed=2)
            f_before = {k: v.clone() for k, v in fresh._forward_map.state_dict().items()}
            actor_before = {k: v.clone() for k, v in fresh._actor.state_dict().items()}
            loaded = frozen_fb.load_frozen_representation(fresh, run)
        self.assertTrue(all(k.startswith(frozen_fb.REPRESENTATION_PREFIXES) for k in loaded))
        for key, value in pretrained._backward_map.state_dict().items():
            self.assertTrue(torch.equal(fresh._backward_map.state_dict()[key], value), key)
        for key, value in pretrained._obs_normalizer.state_dict().items():
            self.assertTrue(torch.equal(fresh._obs_normalizer.state_dict()[key], value), key)
        for key, value in f_before.items():
            self.assertTrue(torch.equal(fresh._forward_map.state_dict()[key], value), f"F changed: {key}")
        for key, value in actor_before.items():
            self.assertTrue(torch.equal(fresh._actor.state_dict()[key], value), f"actor changed: {key}")

    def test_loaded_b_reproduces_pretrained_embeddings(self) -> None:
        pretrained = _tiny_fb_model(seed=1)
        _perturb(pretrained)
        obs = {
            "state": torch.randn(7, 5),
            "privileged_state": torch.randn(7, 6),
            "last_action": torch.randn(7, 3),
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(pretrained, Path(tmp) / "run")
            fresh = _tiny_fb_model(seed=2)
            before = fresh.backward_map(obs)
            frozen_fb.load_frozen_representation(fresh, run)
        expected = pretrained.backward_map(obs)
        self.assertFalse(torch.allclose(before, expected))
        self.assertTrue(torch.allclose(fresh.backward_map(obs), expected, atol=1e-6))

    def test_freeze_stops_gradients_and_statistics_updates(self) -> None:
        pretrained = _tiny_fb_model(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(pretrained, Path(tmp) / "run")
            fresh = _tiny_fb_model(seed=2, prepare_for_train=True)
            frozen_fb.load_frozen_representation(fresh, run)
        for module in (fresh._backward_map, fresh._target_backward_map, fresh._obs_normalizer):
            self.assertFalse(module.training)
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
        # train(True) on the whole model re-enables BatchNorm statistics; freezing must be re-applicable
        fresh.train(True)
        frozen_fb.freeze_representation(fresh)
        before = {k: v.clone() for k, v in fresh._obs_normalizer.state_dict().items()}
        fresh._obs_normalizer(
            {"state": torch.randn(32, 5) * 9 + 3, "privileged_state": torch.randn(32, 6), "last_action": torch.randn(32, 3)}
        )
        for key, value in fresh._obs_normalizer.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), f"normalizer statistic moved: {key}")

    def test_target_b_starts_equal_to_loaded_b(self) -> None:
        pretrained = _tiny_fb_model(seed=1)
        _perturb(pretrained)
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(pretrained, Path(tmp) / "run")
            fresh = _tiny_fb_model(seed=2, prepare_for_train=True)
            frozen_fb.load_frozen_representation(fresh, run)
        for key, value in fresh._backward_map.state_dict().items():
            self.assertTrue(torch.equal(fresh._target_backward_map.state_dict()[key], value), key)

    def test_architecture_mismatch_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _save_pretrained(_tiny_fb_model(state_dim=5, z_dim=8), Path(tmp) / "run")
            for kwargs in ({"state_dim": 7}, {"z_dim": 12}):
                with self.subTest(**kwargs):
                    model = _tiny_fb_model(seed=3, **kwargs)
                    with self.assertRaisesRegex(RuntimeError, "does not match"):
                        frozen_fb.load_frozen_representation(model, run)

    def test_non_fb_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "model").mkdir(parents=True)
            safetensors.torch.save_file({"_actor.w": torch.zeros(2)}, root / "model" / "model.safetensors")
            (root / "model" / "config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "_backward_map"):
                frozen_fb.load_representation_state(root)


class ExpertLatentEncodingTest(unittest.TestCase):
    def _observations(self, length: int):
        gen = torch.Generator().manual_seed(7)
        return {
            "state": torch.randn(length, 5, generator=gen),
            "privileged_state": torch.randn(length, 6, generator=gen),
            "last_action": torch.zeros(length, 3),
        }

    def test_matches_tracking_inference_including_shortened_tail_windows(self) -> None:
        for length, seq_length in ((20, 8), (9, 8), (8, 1), (30, 5)):
            with self.subTest(length=length, seq_length=seq_length):
                model = _tiny_fb_model(seed=4, seq_length=seq_length)
                observations = self._observations(length)
                expected = model.tracking_inference(observations)
                got = frozen_fb.encode_expert_latents(model, observations, seq_length=seq_length)
                self.assertEqual(tuple(got.shape), (length, 8))
                self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_chunked_encoding_matches_single_pass_and_latents_are_on_the_sphere(self) -> None:
        model = _tiny_fb_model(seed=4)
        observations = self._observations(37)
        whole = frozen_fb.encode_expert_latents(model, observations, seq_length=8, batch_size=1000)
        chunked = frozen_fb.encode_expert_latents(model, observations, seq_length=8, batch_size=5)
        self.assertTrue(torch.allclose(whole, chunked, atol=1e-5))
        self.assertTrue(torch.allclose(whole.norm(dim=-1), torch.full((37,), math.sqrt(8)), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
