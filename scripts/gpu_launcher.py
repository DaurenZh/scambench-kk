"""Run GPU scripts with persistent logs and terminate the child on notebook stop."""
import datetime
import os
import pathlib
import subprocess
import sys


def run_logged(script, *arguments, cwd=None):
    root = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd)
    values = list(map(str, arguments))
    destination = root / "launcher_logs"
    if "--output" in values:
        destination = pathlib.Path(values[values.index("--output") + 1])
        if not destination.is_absolute():
            destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    logfile = destination / "console.log"
    command = [sys.executable, "-u", str(root / script), *values]
    print(f"Persistent log: {logfile}", flush=True)
    with logfile.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n--- {datetime.datetime.now().isoformat()} {values!r} ---\n")
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.write("\nNotebook interrupted; child stopped. Resume from a completed checkpoint.\n")
            raise
        finally:
            process.stdout.close()
        if status:
            raise RuntimeError(f"Training exited with code {status}. The full output is saved in {logfile}")


def latest_checkpoint(folder):
    """Ignore checkpoint folders without the state and model needed for recovery."""
    folder = pathlib.Path(folder)
    candidates = []
    for path in folder.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        complete = all((path / f).exists() for f in ["trainer_state.json", "optimizer.pt", "scheduler.pt"])
        model = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        if complete and model:
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"No complete training checkpoint in {folder}; smoke checkpoints cannot recover a full run.")
    return max(candidates)[1]
