from __future__ import annotations

import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from humanoidverse.train import DEFAULT_WORK_DIR, _new_timestamped_run_dir, _select_device_and_rank


class TrainingRunDirectoryTest(unittest.TestCase):
    def test_default_parent_is_bfm_zero_seed(self) -> None:
        self.assertEqual(DEFAULT_WORK_DIR, "runs/bfm_zero_seed")

    def test_timestamped_run_directories_are_unique(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            parent = Path(temporary_dir) / "bfm_zero_seed"
            first = _new_timestamped_run_dir(parent, "20260802_120000")
            second = _new_timestamped_run_dir(parent, "20260802_120000")

            self.assertEqual(first, parent / "20260802_120000")
            self.assertEqual(second, parent / "20260802_120000_01")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_distributed_rank_is_used_without_cuda_visible_devices(self) -> None:
        distributed_env = {
            "LOCAL_RANK": "3",
            "RANK": "3",
            "WORLD_SIZE": "8",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        with patch.dict(environ, distributed_env, clear=True), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.device_count", return_value=8):
            self.assertEqual(_select_device_and_rank(0), ("cuda:3", 3, 3, 8))
            self.assertEqual(environ["MUJOCO_EGL_DEVICE_ID"], "3")

    def test_incomplete_distributed_environment_fails_instead_of_hanging(self) -> None:
        with patch.dict(environ, {"WORLD_SIZE": "8"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Incomplete distributed environment"):
                _select_device_and_rank(0)

    def test_single_process_ignores_stale_local_rank(self) -> None:
        with patch.dict(environ, {"LOCAL_RANK": "7", "RANK": "0", "WORLD_SIZE": "1"}, clear=True), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.device_count", return_value=1):
            self.assertEqual(_select_device_and_rank(0), ("cuda:0", 0, 0, 1))


if __name__ == "__main__":
    unittest.main()
