"""Check recovery selection and retain errors when a notebook subprocess fails."""
import contextlib
import io
import pathlib
import tempfile
import unittest

from gpu_launcher import latest_checkpoint, run_logged


class LauncherTests(unittest.TestCase):
    def test_selects_latest_complete_checkpoint_instead_of_partial_save(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for number in [1, 50, 100]:
                checkpoint = root / f"checkpoint-{number}"
                checkpoint.mkdir()
                for name in ["model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt"]:
                    if number != 100 or name != "trainer_state.json":
                        (checkpoint / name).touch()
            self.assertEqual(latest_checkpoint(root).name, "checkpoint-50")

    def test_no_checkpoint_does_not_fall_back_to_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint(folder)

    def test_error_output_is_persistent_after_child_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            (root / "fail.py").write_text("import sys\nprint('before failure')\nraise RuntimeError('example failure')\n")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "console.log"):
                    run_logged("fail.py", "--output", "results", cwd=root)
            log = (root / "results/console.log").read_text()
            self.assertIn("before failure", log)
            self.assertIn("RuntimeError: example failure", log)


if __name__ == "__main__":
    unittest.main()
