# Action prediction and conditional diffusion

These experiments extend the existing risk classification task. Implemented and
locally checked; full CUDA results must be collected on the separate GPU machine.

## Decision making as offline action prediction

Predict the source corpus's `decision_class` from conversation messages. Examples
include `engage_legitimate`, `request_verification`, `audit`, and `refuse`. The local
snapshot has 13 action labels after replacing hyphens with underscores. The source
defines this field as the agent's decision category:
[ScamBench dataset card](https://huggingface.co/datasets/shaw/scambench-training).

`decision_class` is a target here. It remains excluded from the model input, along
with `chosen_action`, `response_text`, `explanation`, risk labels and diagnostics.
Action names are an explicit output schema; per-split support is exported. The
Kazakh targets inherit the English source action, without fresh human annotation.

This evaluates agreement with recorded agent decisions. It does not establish the
correctness of the source policy, execute actions, learn interaction rewards, or
measure real-world consequences. Some actions are very rare or absent from the
300-pair evaluation; report class support and per-class scores next to macro-F1.
In the current benchmark, only five actions occur: `audit`, `engage_legitimate`,
`ignore`, `refuse`, and `request_verification`. Perfect predictions therefore score
only 5/13 on fixed-schema macro-F1 when the eight absent classes receive zero F1.
Use the separately reported supported-class macro-F1 to assess those five actions,
and do not claim evaluation coverage of all 13 actions.

Models: end-to-end XLM-R, original TF-IDF classifiers, and the matched dense-feature
linear/MLP/diffusion comparison described below. None receives action metadata as
input. Use `--target action`; `--target risk` preserves three-class classification.

## Diffusion as conditional label generation

[CARD (Han, Zheng and Zhou, NeurIPS 2022)](https://arxiv.org/abs/2206.07275)
combines a conditional mean predictor with denoising diffusion to model a target
distribution. Its [official implementation](https://github.com/XzwHan/CARD) includes
classification using continuous label vectors. This makes it a suitable research
starting point for our categorical targets.

Our adaptation uses word TF-IDF, a 128-dimensional SVD projection, and standardization.
All transformations fit only the selected training rows. A neural classifier supplies
a frozen conditional mean `p(x)`. For a one-hot target `y`, diffuse the residual:

```
r_0 = y - p(x)
r_t = sqrt(alpha_bar_t) * r_0 + sqrt(1 - alpha_bar_t) * noise
```

A conditional network predicts the added noise from text features, `r_t`, `p(x)`
and the timestep. Reverse DDPM sampling reconstructs target vectors; repeated
samples vote for a class. Samples never require the evaluation example's true label.
Sample agreement is an exploratory statistic, not a calibrated probability of safety.

This is an adaptation, not a reproduction of CARD's architectures or reported results.
It uses a cosine schedule, 100 reverse steps, clipped reconstructed one-hot vectors,
and eight samples per example. It generates labels, not synthetic conversations.

The matched linear SVM and MLP use exactly the same SVD features, so the comparison
isolates the diffusion head. Compare against full TF-IDF and fine-tuned XLM-R as
separate systems too: compression may lose useful lexical evidence.

## Protocol and running

Build the updated `data/gpu_bundle.zip` with `scripts/prepare_gpu_bundle.py`. Extract
on the Windows GPU machine and open `06_decisions_and_diffusion.ipynb`. Notebook 05
still runs the original XLM-R risk experiments. Notebook 06 adds action prediction
and diffusion for both targets. It runs short smoke checks before full training.

Both English-only and mixed training are supported. The first run uses mixed data;
freeze settings before repeating with English-only data or additional seeds. Both
use the same held-out EN + silver development families. Silver translations stay
with their source family; the 300 gold pairs never enter training or validation.
Feature fitting and checkpoint selection do not read gold evaluation files.

Keep training, validation and final evaluation separate. Action scores use the fixed
13-class output schema and also report macro-F1 over classes present in the evaluated
set. Risk scores retain the three-class and merged-binary metrics. Inspect missing
training targets, prediction files, class support, and errors before interpreting
aggregate results. The historical benchmark has already informed project choices;
it is not a fresh confirmatory test, and native-speaker review remains outstanding.
