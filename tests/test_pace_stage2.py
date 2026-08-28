from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer, TrajectoryDictBufferMultiDim
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore
from humanoidverse.training.workspace import Workspace, _compute_pace_emd_distribution


def _episode(motion_id: int, length: int = 12) -> dict:
    return {
        "observation": {"state": torch.arange(length, dtype=torch.float32).unsqueeze(-1)},
        "terminated": torch.zeros(length, dtype=torch.bool),
        "truncated": torch.tensor([False] * (length - 1) + [True]),
        "motion_id": torch.full((length,), motion_id, dtype=torch.long),
    }


class PaceStage2Test(unittest.TestCase):
    def test_tracking_eval_uses_narrow_pace_suspension(self) -> None:
        class MotionLib:
            def load_motions_for_training(self, max_num_seqs):
                self.loaded = max_num_seqs

        env = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        env.pace_rollout_suspended = False
        env.is_evaluating = False
        env.num_envs = 2
        env._motion_lib = MotionLib()

        env.suspend_pace_rollout_for_evaluation()
        self.assertTrue(env.pace_rollout_suspended)
        self.assertFalse(env.is_evaluating)
        env.prepare_training_motion_pool()
        self.assertFalse(env.pace_rollout_suspended)
        self.assertFalse(env.is_evaluating)
        self.assertEqual(env._motion_lib.loaded, 2)

    def test_top_ten_percent_has_eighty_percent_sampling_mass(self) -> None:
        metrics = {
            f"motion_{motion_id}": {"motion_id": motion_id, "emd": float(motion_id)}
            for motion_id in range(20)
        }
        result = _compute_pace_emd_distribution(
            metrics,
            list(range(20)),
            challenging_fraction=0.10,
            challenging_sampling_ratio=0.80,
            clamp_min=0.5,
            clamp_max=2.0,
            scale=2.0,
        )
        self.assertEqual(torch.bincount(result["group_ids"], minlength=2).tolist(), [18, 2])
        challenging = result["group_ids"] == 1
        self.assertAlmostEqual(float(result["probabilities"][challenging].sum()), 0.8, places=6)

    def test_next_frame_expert_window_uses_real_buffer_bounds(self) -> None:
        buffer = TrajectoryDictBuffer(episodes=[_episode(0)], seq_length=8, device="cpu")
        window = buffer.observation_sequence(0, start_step=2, num_steps=8, next_offset=1)
        self.assertEqual(window["state"].flatten().tolist(), list(map(float, range(3, 11))))
        with self.assertRaises(IndexError):
            buffer.observation_sequence(0, start_step=4, num_steps=8, next_offset=1)

    def test_motion_start_bounds_use_expert_lengths_not_motion_lib_rounding(self) -> None:
        class MotionLib:
            _emd_group_ids = torch.tensor([0, 1])

            @staticmethod
            def sample_motions_from_groups(groups):
                return torch.arange(len(groups), dtype=torch.long)

            @staticmethod
            def get_motion_length(motion_ids):
                return torch.ones_like(motion_ids, dtype=torch.float32)

            @staticmethod
            def global_motion_ids(motion_ids):
                return motion_ids + 2

        env = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        env.device = "cpu"
        env.pace_enabled = True
        env.pace_rollout_suspended = False
        env.is_evaluating = False
        env._motion_lib = MotionLib()
        env.pace_acquisition_mask = torch.tensor([True, False])
        env.pace_expert_motion_lengths = torch.tensor([9, 9, 500, 301])
        env.pace_seq_length = 8
        env.pace_expert_rollout_max_steps = 250
        env.motion_ids = torch.zeros(2, dtype=torch.long)
        env.motion_len = torch.zeros(2)
        env.motion_start_times = torch.zeros(2)
        env.pace_motion_start_steps = torch.zeros(2, dtype=torch.long)
        env.pace_segment_lengths = torch.zeros(2, dtype=torch.long)
        env.dt = 0.02

        torch.manual_seed(4)
        env._resample_motion_time_and_ids(torch.arange(2))

        self.assertEqual(env.pace_segment_lengths.tolist(), [499, 0])
        self.assertEqual(int(env.pace_motion_start_steps[0]), 0)
        self.assertLessEqual(int(env.pace_motion_start_steps[1]), 301 - 8 - 1)

    def test_eval_reset_discards_partial_validity_window(self) -> None:
        workspace = Workspace.__new__(Workspace)
        workspace.cfg = SimpleNamespace(pace=object())
        workspace._pace_window_step = 7
        workspace._pace_valid_acquisition = 11
        workspace._pace_valid_consolidation = 12
        workspace._pace_total_acquisition = 13
        workspace._pace_total_consolidation = 14
        workspace._pace_alive = np.array([False, True, False], dtype=bool)

        workspace._discard_partial_pace_window()

        self.assertEqual(workspace._pace_window_step, 0)
        self.assertEqual(workspace._pace_valid_acquisition, 0)
        self.assertEqual(workspace._pace_valid_consolidation, 0)
        self.assertEqual(workspace._pace_total_acquisition, 0)
        self.assertEqual(workspace._pace_total_consolidation, 0)
        self.assertTrue(workspace._pace_alive.all())

    def test_valid_sampler_uses_rank_safe_replacement_fallback(self) -> None:
        buffer = TrajectoryDictBufferMultiDim.__new__(TrajectoryDictBufferMultiDim)
        buffer.storage = {"pace_valid": torch.ones(1, dtype=torch.bool)}

        def sparse_sample(batch_size):
            valid = torch.zeros(batch_size, 1, dtype=torch.bool)
            valid[0] = True
            return {"pace_valid": valid, "value": torch.arange(batch_size).unsqueeze(-1)}

        buffer.sample = sparse_sample
        with self.assertWarnsRegex(RuntimeWarning, "filling the batch with replacement"):
            result = buffer.sample_valid(32)
        self.assertEqual(tuple(result["value"].shape), (32, 1))
        self.assertTrue(result["pace_valid"].all())


if __name__ == "__main__":
    unittest.main()
