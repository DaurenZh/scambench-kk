# Running on a GTX 1650 Ti with 4 GB VRAM

The old failure cannot be diagnosed without its traceback. The most recent run
was stopped manually; only a smoke-test checkpoint was found. It cannot recover
the English experiment. Keep the old folder, but use a fresh extracted folder for
the updated notebooks.

## Start here

1. Extract the updated ZIP on Windows and open the extracted folder in VS Code.
2. Select the existing CUDA PyTorch kernel and open notebook 05.
3. Run setup and the smoke check. Do not repeat a smoke check in a directory that
   already contains its checkpoints; select a fresh output directory if necessary.
4. If the check completes, run English training, then mixed training. Keep Windows
   awake and plugged in. A successful smoke check does not predict total runtime.
5. Run evaluation only after training finishes. Then continue with notebook 06.

The notebook configuration is batch size 1, accumulation 32, 256-token head/tail
inputs, and Adafactor. Both language conditions use the same configuration. This
is a new experiment compared with the previous AdamW/512-token setup: optimizer
and context length changes must be reported with the results. The command-line
runner retains its original optimizer/context defaults for explicit use elsewhere.

Adafactor is an alternative optimizer with reduced optimizer-state memory:
[official Transformers optimization documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor).
This configuration is intended for a small GPU, but it has not been tested on the
user's GTX 1650 Ti. If the smoke check fails, send `console.log` before another
full run. The smaller diffusion/MLP experiments in notebook 06 are independent of
XLM-R training and can also be run separately.

## Checkpoints, stopping and resuming

XLM-R writes a full recovery checkpoint after optimizer update 1, then every 50
updates and at epoch boundaries. This interval is measured in optimizer updates,
not seconds. The Trainer retains the best checkpoint and a recent recovery
checkpoint. Checkpoints take disk space and time; leave roughly 15 GB free for
the model cache and several experiments.

To stop, interrupt the running notebook cell. The launcher terminates its training
subprocess; unfinished work since the last completed checkpoint is lost. It does
not manufacture a checkpoint during termination. Check that GPU usage subsides
before starting another process. Closing VS Code or shutting down Windows does
not give the launcher an opportunity to handle the interruption.

For notebook 05, replace the English training cell with:

```python
train_variant("en", resume=True)
```

For mixed training use `train_variant("mixed", resume=True)`. For notebook 06's
XLM-R action run use `train_action(resume=True)`. Run setup first and leave the
training settings unchanged. Recovery uses the latest checkpoint with model,
optimizer, scheduler and trainer-state files. Do not use a smoke checkpoint to
resume a full run. Diffusion training currently has no resume command.

## Logs

Each subprocess writes its output and tracebacks to `<run folder>/console.log`.
Notebook 05 uses `gpu_results_4gb/`; notebook 06 uses
`decision_diffusion_results_4gb/`. A failed command points to the log file. The
result summary ZIPs now include these logs along with metrics and settings.

Local verification used a tiny random XLM-R on CPU: first-update checkpoint,
intentional interruption, optimizer-state recovery and final model saving all
passed. This checks the recovery mechanism, not the full model's GPU memory use
or classification quality.
