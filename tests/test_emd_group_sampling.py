from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluation
from humanoidverse.agents.fb.agent import FBAgent
from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.training.workspace import _compute_emd_group_distribution, _log_emd_group_stats
from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase


def _metrics(values: list[float]) -> dict[str, dict[str, float | int | str]]:
    return {
        f"motion_{motion_id}": {
            "motion_id": motion_id,
            "motion_file": f"motion_{motion_id}",
            "emd": emd,
        }
        for motion_id, emd in enumerate(values)
    }


class EmdGroupDistributionTest(unittest.TestCase):
    def test_boundaries_group_mass_and_current_weight_formula(self) -> None:
        result = _compute_emd_group_distribution(
            _metrics([0.1, 0.4, 0.5, 1.0, 2.0, 3.0]),
            list(range(6)),
            low_threshold=0.5,
            high_threshold=2.0,
            scale=2.0,
            high_group_sampling_ratio=0.05,
        )

        self.assertEqual(result["group_ids"].tolist(), [0, 0, 0, 0, 1, 1])
        torch.testing.assert_close(result["group_weights"], torch.tensor([0.95, 0.05]))
        torch.testing.assert_close(
            result["probabilities"],
            torch.tensor([0.19, 0.19, 0.19, 0.38, 0.025, 0.025]),
        )
        self.assertAlmostEqual(
            sum(
                result["stats"][f"eval/emd_sampling/{name}/sampling_ratio"]
                for name in ("non_high", "high")
            ),
            1.0,
        )
        self.assertAlmostEqual(result["stats"]["eval/emd_sampling/high/data_ratio"], 1 / 3)

    def test_empty_groups_log_zero(self) -> None:
        result = _compute_emd_group_distribution(
            _metrics([0.1, 0.2]),
            [0, 1],
            low_threshold=0.5,
            high_threshold=2.0,
            scale=2.0,
            high_group_sampling_ratio=0.05,
        )
        self.assertEqual(result["stats"]["eval/emd_sampling/high/count"], 0)
        self.assertEqual(result["stats"]["eval/emd_sampling/high/data_ratio"], 0.0)
        self.assertEqual(result["stats"]["eval/emd_sampling/high/sampling_ratio"], 0.0)
        self.assertEqual(result["stats"]["eval/emd_sampling/high/mean_emd"], 0.0)
        self.assertEqual(result["stats"]["eval/emd_sampling/non_high/sampling_ratio"], 1.0)
        torch.testing.assert_close(result["probabilities"], torch.tensor([0.5, 0.5]))

    def test_invalid_or_incomplete_eval_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite EMD"):
            _compute_emd_group_distribution(
                _metrics([float("nan")]),
                [0],
                low_threshold=0.5,
                high_threshold=2.0,
                scale=2.0,
                high_group_sampling_ratio=0.05,
            )
        with self.assertRaisesRegex(ValueError, "do not match expert data"):
            _compute_emd_group_distribution(
                _metrics([0.5]),
                [0, 1],
                low_threshold=0.5,
                high_threshold=2.0,
                scale=2.0,
                high_group_sampling_ratio=0.05,
            )
        with self.assertRaisesRegex(ValueError, "0 <= low < high"):
            _compute_emd_group_distribution(
                _metrics([0.5]),
                [0],
                low_threshold=2.0,
                high_threshold=0.5,
                scale=2.0,
                high_group_sampling_ratio=0.05,
            )

    def test_wandb_and_tensorboard_receive_identical_group_stats(self) -> None:
        class Writer:
            def __init__(self):
                self.values = {}
                self.flushed = False

            def add_scalar(self, key, value, step):
                self.values[key] = (value, step)

            def flush(self):
                self.flushed = True

        stats = {
            "eval/emd_sampling/non_high/data_ratio": 0.75,
            "eval/emd_sampling/non_high/sampling_ratio": 0.95,
            "eval/emd_sampling/non_high/count": 5,
            "eval/emd_sampling/non_high/mean_emd": 0.2,
        }
        writer = Writer()
        with patch("humanoidverse.training.workspace.wandb.log") as wandb_log:
            _log_emd_group_stats(stats, timestep=3200, use_wandb=True, writer=writer)
        wandb_log.assert_called_once_with(stats, step=3200)
        self.assertEqual(writer.values, {key: (value, 3200) for key, value in stats.items()})
        self.assertTrue(writer.flushed)


class EmdGroupMotionPoolTest(unittest.TestCase):
    def _motion_lib(self) -> MotionLibBase:
        motion_lib = MotionLibBase.__new__(MotionLibBase)
        motion_lib._device = "cpu"
        motion_lib._num_unique_motions = 6
        motion_lib._motion_source_names = ["complete_dataset"]
        motion_lib._curr_motion_ids = None
        motion_lib.all_motions_loaded = True
        return motion_lib

    def test_pool_uses_exact_group_counts_and_maximizes_coverage(self) -> None:
        motion_lib = self._motion_lib()
        probabilities = torch.tensor([0.19, 0.19, 0.19, 0.38, 0.025, 0.025])
        group_ids = torch.tensor([0, 0, 0, 0, 1, 1])
        group_weights = torch.tensor([0.95, 0.05])
        motion_lib.set_emd_group_sampling(probabilities, group_ids, group_weights)

        torch.manual_seed(7)
        selected = motion_lib._sample_indices_with_emd_groups(12, exact_counts=True)
        selected_groups = group_ids[selected]
        self.assertEqual(torch.bincount(selected_groups, minlength=2).tolist(), [11, 1])
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(torch.unique(selected)), 5)

        tie_counts = motion_lib._source_counts(2, group_weights, exact_counts=True)
        self.assertEqual(tie_counts.tolist(), [2, 0])

    def test_each_nonempty_group_gets_a_pool_slot_when_capacity_allows(self) -> None:
        motion_lib = self._motion_lib()
        motion_lib.set_emd_group_sampling(
            torch.tensor([0.24975, 0.24975, 0.24975, 0.24975, 0.0005, 0.0005]),
            torch.tensor([0, 0, 0, 0, 1, 1]),
            torch.tensor([0.999, 0.001]),
        )
        selected = motion_lib._sample_indices_with_emd_groups(2, exact_counts=True)
        selected_groups = motion_lib._emd_group_ids[selected]
        self.assertEqual(torch.bincount(selected_groups, minlength=2).tolist(), [1, 1])

    def test_reset_sampling_is_uniform_over_loaded_slots(self) -> None:
        motion_lib = self._motion_lib()
        motion_lib.set_emd_group_sampling(
            torch.tensor([0.1, 0.1, 0.15, 0.6, 0.025, 0.025]),
            torch.tensor([0, 0, 0, 0, 1, 1]),
            torch.tensor([0.95, 0.05]),
        )
        motion_lib._curr_motion_ids = torch.tensor([3, 3, 0, 2])
        samples = motion_lib.sample_motions(1000)
        self.assertTrue(torch.all((samples >= 0) & (samples < 4)))
        counts = torch.bincount(samples, minlength=4).float()
        self.assertLess(float((counts / counts.sum() - 0.25).abs().max()), 0.06)


class EmdGroupConfigTest(unittest.TestCase):
    def test_build_config_exposes_thresholds_and_cpu_expert_buffer(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/emd_group_config_test",
            num_envs=2,
            num_env_steps=4,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            emd_group_low_threshold=0.25,
            emd_group_high_threshold=1.75,
        )
        self.assertEqual(cfg.prioritization_min_val, 0.25)
        self.assertEqual(cfg.prioritization_max_val, 1.75)
        self.assertEqual(cfg.prioritization_high_group_ratio, 0.05)
        self.assertEqual(cfg.expert_buffer_device, "cpu")


class CpuExpertBufferTest(unittest.TestCase):
    def test_rollout_tracking_observation_moves_to_agent_device(self) -> None:
        class ExpertSlicer:
            def sample(self, batch_size, seq_length):
                self.sample_args = (batch_size, seq_length)
                return {"next": {"observation": {"state": torch.ones(batch_size, 3)}}}

        class Model:
            def backward_map(self, observation):
                self.observation_device = observation["state"].device.type
                return torch.ones(observation["state"].shape[0], 4, device=observation["state"].device)

            @staticmethod
            def project_z(z):
                return z

        expert_slicer = ExpertSlicer()
        model = Model()
        agent = SimpleNamespace(
            device="meta",
            _model=model,
            cfg=SimpleNamespace(model=SimpleNamespace(seq_length=2)),
        )

        z = FBAgent._sample_tracking_z(
            agent,
            {"expert_slicer": expert_slicer},
            batch_dim=3,
            traj_length=2,
        )

        self.assertEqual(expert_slicer.sample_args, (6, 2))
        self.assertEqual(model.observation_device, "meta")
        self.assertEqual(tuple(z.shape), (3, 2, 4))


class ChunkedEvaluationTest(unittest.TestCase):
    def test_complete_eval_loads_global_motion_ids_in_num_env_chunks(self) -> None:
        class MotionLib:
            _num_unique_motions = 5
            all_motions_loaded = True

            def __init__(self):
                self.loaded = []

            def load_motions(self, *, random_sample, motion_ids):
                self.loaded.append(list(motion_ids))

        class CoreEnv:
            num_envs = 2

            def __init__(self):
                self._motion_lib = MotionLib()

        env = SimpleNamespace(_env=CoreEnv())
        evaluation = HumanoidVerseMjlabTrackingEvaluation.__new__(HumanoidVerseMjlabTrackingEvaluation)
        evaluation.cfg = SimpleNamespace(
            env=None,
            num_envs=2,
            n_episodes_per_motion=1,
            disable_tqdm=True,
            include_results_from_all_envs=False,
        )

        def fake_worker(inputs, **_kwargs):
            _local_ids, global_ids, _agent = inputs
            return {
                f"motion_{motion_id}": {"motion_id": motion_id, "emd": float(motion_id)}
                for motion_id in global_ids
            }

        with patch(
            "humanoidverse.agents.evaluations.humanoidverse_mjlab._async_tracking_worker",
            side_effect=fake_worker,
        ):
            metrics, _aggregate = evaluation.run(timestep=10, agent_or_model=object(), logger=None, env=env)

        self.assertEqual(env._env._motion_lib.loaded, [[0, 1], [2, 3], [4]])
        self.assertEqual(sorted(metric["motion_id"] for metric in metrics.values()), list(range(5)))


if __name__ == "__main__":
    unittest.main()
