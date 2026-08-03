from __future__ import annotations

import unittest

import numpy as np
import torch

from humanoidverse.training.workspace import (
    _as_python_scalar,
    _build_grouped_metrics,
    _iter_tracked_group_metrics,
    _log_grouped_tensorboard,
    _render_grouped_train_log,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float | int, int]] = []
        self.flush_count = 0

    def add_scalar(self, tag: str, value: float | int, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1


class TrainingLoggingTest(unittest.TestCase):
    def test_scalar_conversion_matches_wandb_and_tensorboard_inputs(self) -> None:
        self.assertEqual(_as_python_scalar(torch.tensor(3)), 3)
        self.assertEqual(_as_python_scalar(np.asarray(2.5, dtype=np.float32)), 2.5)
        self.assertEqual(_as_python_scalar(np.bool_(True)), 1)
        self.assertIsNone(_as_python_scalar("local_loss_average"))
        self.assertIsNone(_as_python_scalar(torch.ones(2)))

    def test_metrics_are_grouped_and_non_scalars_are_excluded(self) -> None:
        grouped = _build_grouped_metrics(
            {
                "B": torch.tensor(1.25),
                "tldr_te_loss": np.float32(2.5),
                "distributed/world_size": 8,
                "distributed/loss_mode": "local_loss_average",
                "new_metric": 4.0,
            }
        )

        self.assertEqual(list(grouped), ["FB", "TeCH", "Distributed", "Other"])
        emitted = {key: value for _, key, value in _iter_tracked_group_metrics(grouped)}
        self.assertEqual(emitted["B"], 1.25)
        self.assertEqual(emitted["tldr_te_loss"], 2.5)
        self.assertEqual(emitted["distributed/world_size"], 8)
        self.assertNotIn("new_metric", emitted)
        self.assertNotIn("distributed/loss_mode", emitted)

    def test_grouped_terminal_log_follows_reference_layout(self) -> None:
        grouped = _build_grouped_metrics({"B": 0.125, "actor_loss": -2.0})
        rendered = _render_grouped_train_log(
            grouped_metrics=grouped,
            timestep=8192,
            iteration=1,
            max_iterations=100,
            steps_per_second=4096.0,
            collection_time=1.0,
            learning_time=2.0,
            elapsed_time=3.0,
            eta_seconds=30.0,
        )

        self.assertIn("Learning iteration 1/100", rendered)
        self.assertIn("FB/B: 0.125", rendered)
        self.assertIn("Actor/actor_loss: -2", rendered)
        self.assertIn("Total timesteps: 8192", rendered)
        self.assertIn("ETA: 00:00:30", rendered)

    def test_tensorboard_receives_the_same_tracked_metric_types_as_wandb(self) -> None:
        writer = _RecordingWriter()
        grouped = _build_grouped_metrics(
            {
                "B": 0.5,
                "aux_rew/penalty_torques": -0.25,
                "distributed/world_size": 8,
            }
        )
        _log_grouped_tensorboard(
            writer,
            grouped,
            8192,
            steps_per_second=4096.0,
            collection_time=1.0,
            learning_time=2.0,
            elapsed_time=3.0,
            eta_seconds=30.0,
            iteration=1,
        )

        tags = {tag for tag, _, _ in writer.scalars}
        self.assertIn("train/fb/B", tags)
        self.assertIn("train/aux_rewards/penalty_torques", tags)
        self.assertIn("train/distributed/world_size", tags)
        self.assertTrue(all(step == 8192 for _, _, step in writer.scalars))
        self.assertEqual(writer.flush_count, 1)


if __name__ == "__main__":
    unittest.main()
