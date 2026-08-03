from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from humanoidverse.train import DEFAULT_WORK_DIR, _new_timestamped_run_dir


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


if __name__ == "__main__":
    unittest.main()
