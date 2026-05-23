# Pet Breed Classifier — A Six-Model Tournament on Oxford-IIIT Pet

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/onurcanmemis/Dog_Breed_Classifier)

A small, end-to-end study in fine-grained visual classification: take a 37-class
pet-breed problem, run six architectures against it under identical training
constraints, and ship the winner behind an explainable FastAPI web app
(containerised for Hugging Face Spaces).

> Evolved from a cats-vs-dogs binary classifier. Binary cats-vs-dogs is solved
> — fine-grained breed identification is not, and is a closer proxy for the
> kind of vision problems that actually matter in production.

---

## Quick start

```bash
uv sync                                            # install deps (incl. tensorflow-metal on Apple Silicon)
uv run python download_dataset.py                  # fetch & organise Oxford-IIIT Pet (~800 MB)
uv run python model_experiments/train_compare.py   # full 6-model tournament
uv run uvicorn app:app --reload --port 7860        # serve the winner; open http://localhost:7860
```

`train_compare.py` writes results incrementally and is safe to interrupt — completed
models persist; the next run replays the rest. The winning `.keras` artefact is copied
to `./cnn_model.keras` and `app.py` picks it up automatically.

---

## 1. Introduction

This project trains and compares six convolutional architectures on the
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset (37 cat
and dog breeds, ~7,400 images) and deploys the best of them as a FastAPI web
app with Grad-CAM explanations, containerised for Hugging Face Spaces.

The tournament structure exists to answer a concrete engineering question: on
a small fine-grained dataset, what is actually worth shipping — a custom CNN,
a pretrained backbone with a frozen body, or a pretrained backbone with the
top layers fine-tuned? The output is not just a number but a side-by-side
view of accuracy, model size, and inference latency.

## 2. Business Context

Automated breed identification has real applications:

- **Pet insurance** — automated breed verification on uploaded photos at
  policy enrolment.
- **Veterinary triage** — breed-specific health risk scoring before a clinic
  visit (e.g. brachycephalic airway syndrome screening in flat-faced breeds).
- **Adoption platforms** — automated breed tagging so unstructured uploads
  become searchable inventory.

The Grad-CAM explainability layer addresses the regulatory and trust
requirement increasingly placed on automated visual decisions: a clinician or
underwriter can see _which pixels_ produced the prediction, not just the
prediction itself.

## 3. Hypothesis

1. **Transfer learning will beat training from scratch.** Both MobileNetV2
   and EfficientNetB0 pretrained on ImageNet will significantly outperform
   the custom CNNs, because ~7,400 images across 37 fine-grained classes is
   too small for a deep network to learn robust low-level visual features from
   scratch.
2. **Fine-tuning will win, but by a narrow margin.** Unfreezing the top 20 %
   of EfficientNetB0 and continuing training at a low learning rate will yield
   the highest absolute accuracy, but the marginal gain over the frozen
   variant will be modest — ImageNet features already encode strong
   pet-relevant representations.
3. **There will be a meaningful speed/accuracy trade-off.** Custom CNNs will
   be fastest at inference but least accurate; EfficientNetB0 fine-tuned will
   be most accurate but slowest. The deployable model isn't necessarily the
   most accurate one — it's the one with the best speed/accuracy tradeoff
   closest to the product's latency budget.

## 4. Methodology

### 4.1 Dataset

The Oxford-IIIT Pet dataset (37 classes, ~200 images per breed) is downloaded
on demand by `download_dataset.py` and organised into per-breed folders using
the official `trainval` / `test` split:

```
dataset_oxford_pets/
├── trainval/<breed>/   # ~3,680 images — train_compare.py carves 20 % off as val
└── test/<breed>/       # ~3,669 images — touched once, only on the tournament winner
```

Class encoding is alphabetical-by-folder-name; cat breeds carry a capitalised
prefix and dog breeds lowercase, which is preserved from the dataset's own
filename convention.

**Class distribution.** Oxford-IIIT Pet is *approximately* — not perfectly —
balanced. `train_compare.py` counts images per class on startup, logs
`min` / `max` / `mean`, and flags any breed below 150 images. The full
distribution is saved as a horizontal bar chart at
`model_experiments/plots/class_distribution.png`. Because the imbalance is
real but mild, **macro-averaged F1** is treated as the primary classification
metric — overall accuracy is reported alongside but is inflated by the more
populous classes. Macro F1 weights every breed equally regardless of size.

### 4.2 Models

| # | Model | Type | Preprocessing | Max epochs |
|---|---|---|---|---|
| 1 | Baseline CNN | From scratch | `image / 255` | 25 |
| 2 | Deeper CNN | From scratch (BN + dropout) | `image / 255` | 25 |
| 3 | MobileNetV2 (frozen) | Pretrained, head only | `mobilenet_v2.preprocess_input` | 15 |
| 4 | MobileNetV2 (fine-tuned) | Pretrained, top 20 % unfrozen | `mobilenet_v2.preprocess_input` | 10 |
| 5 | EfficientNetB0 (frozen) | Pretrained, head only | `efficientnet.preprocess_input` | 15 |
| 6 | EfficientNetB0 (fine-tuned) | Pretrained, top 20 % unfrozen | `efficientnet.preprocess_input` | 10 |

Each model has its own factory file under `model_experiments/models/`; the
tournament runner imports them by module and treats them uniformly.

#### Why not ResNet50?

ResNet50 was considered and dropped. At ~100 MB serialised vs ~20 MB for
EfficientNetB0, and with no clear accuracy advantage on a dataset this size,
including it would have inflated deployment cost (and Hugging Face Space
storage) without changing the conclusion. Pragmatic deployment beats
exhaustive benchmarking.

### 4.3 Training setup

- **Input size**: 224 × 224 × 3 — the native resolution the pretrained
  backbones were designed for.
- **Batch size**: 32 — tuned for an M1 with 16 GB of unified memory.
- **Optimiser**: Adam, learning rate `1e-3` for head training, `1e-5` for
  fine-tuning. The drop is critical: at any meaningfully higher fine-tune LR
  the pretrained weights are destroyed in the first epoch.
- **Early stopping**: patience 5, monitor `val_loss`, `restore_best_weights=True`,
  applied uniformly across all six models — no per-model hand-tuning.
- **Validation split**: stratified 80 / 20 from `trainval/`, seeded for
  reproducibility. The test set is never touched until the final evaluation.
- **Random seed**: 42 (`random`, `numpy`, `tensorflow`, `PYTHONHASHSEED`).

### 4.4 Three runtime optimisations

1. **Mixed precision (`mixed_float16`)** — global policy set before any model
   is built. All output `Dense` layers are explicitly `dtype='float32'` to
   keep the softmax / cross-entropy numerically stable.
2. **`tf.data` cached in RAM** — images are decoded and resized once into
   uint8 (about 1 GB total), then cached. Augmentation and model-specific
   preprocessing layer on top via cheap `.map` calls that run from cache each
   epoch. Disk reads happen only during epoch 1.
3. **`tensorflow-metal`** — Apple's Metal-backed TF plug-in is added as a
   platform-conditional dependency for Apple Silicon GPU acceleration.

### 4.5 Data augmentation

Applied to training data only, never to validation or test:

| Aug | Reason |
|---|---|
| Horizontal flip ✅ | A mirrored pet is still the same breed. |
| Random rotation ±15° ✅ | Real pet photos aren't perfectly upright. |
| Brightness ±20% ✅ | Different lighting conditions. |
| Zoom up to 10% ✅ | Animals at different camera distances. |
| Vertical flip ❌ | Upside-down animals are unrealistic. |
| Aggressive shear ❌ | Destroys the fine-grained features we're trying to learn. |

### 4.6 Metrics & logging

For every model the tournament logs validation accuracy / loss / precision /
recall / F1 (per class and macro on the validation set), trainable parameter
count, training time, median single-image inference latency over 500 test
images, model file size, the training history, and the early-stopping
outcome. Results are written incrementally to
`model_experiments/results/results_YYYY-MM-DD.json` so a mid-run crash never
throws away the models that already finished.

**Winner selection is by `val_accuracy`, not test accuracy.** The test set
is touched *exactly once* — on the winning architecture, after selection —
and the resulting `test_accuracy` / `test_f1_macro` is the single
real-world performance figure reported. Selecting by test accuracy would
turn the test set into a selection criterion (validation by another name)
and inflate the headline number.

The winning `.keras` artefact is then copied to `./cnn_model.keras` for the
app to consume.

## 5. Results

> Numbers below are populated by `train_compare.py`. Re-run the tournament to
> refresh them; the canonical source of truth is
> `model_experiments/results/results_<date>.json`.

### 5.1 Headline table

Per-model rows report **validation** numbers. The test set is touched only
on the eventual winner; its `test_accuracy` / `test_f1_macro` are reported
separately in the row below the table.

| Model | Val acc | Val F1 (macro) | Params | Size (MB) | Inference (ms, median) |
|---|---|---|---|---|---|
| baseline_cnn | — | — | — | — | — |
| deeper_cnn | — | — | — | — | — |
| mobilenet_frozen | — | — | — | — | — |
| mobilenet_finetune | — | — | — | — | — |
| efficientnetb0_frozen | — | — | — | — | — |
| efficientnetb0_finetune | — | — | — | — | — |

| Winner | Test acc | Test F1 (macro) |
|---|---|---|
| — | — | — |

### 5.2 Learning curves

One PNG per model under
`model_experiments/plots/learning_curves_<date>/<model>.png`.

### 5.3 Speed vs accuracy

Saved to `model_experiments/plots/speed_vs_accuracy_<date>.png` — a scatter
plot of median per-image inference latency (x, measured over 500 single-image
predictions after a GPU warm-up) against **validation accuracy** (y), one
point per model. Validation accuracy rather than test accuracy because only
the winner is evaluated on the test set; val_accuracy is the metric that
exists for all six.

The deployable model is the one nearest the top-left _given the product's
latency budget_; chasing the absolute top isn't always justified. (This is a
straightforward scatter plot, not a formal Pareto-frontier computation —
"speed/accuracy tradeoff" is the cleaner description of what's being shown.)

## 6. Conclusions

Populate after the tournament runs. The structure to fill in:

- Was hypothesis 1 (transfer learning > scratch) confirmed?
- How large was the fine-tune vs frozen gap, and was hypothesis 2 borne out?
- Where does the deployed winner sit on the speed/accuracy frontier, and what
  product context justifies that choice?

## 7. Future work

- **ResNet50** — include for completeness once a 100 MB-class artefact is
  acceptable in deployment; rule it in/out _from data_, not assumption.
- **Failure-mode analysis** — cluster misclassified images by colour
  histogram, sharpness, and aspect-ratio to surface systematic weaknesses
  (e.g. is the model confusing all dark-coated short-haired breeds?).
- **Input-size sweep** — does 299 × 299 or 384 × 384 fine-tuning unlock more
  accuracy at the cost of latency, and how does the speed/accuracy tradeoff
  shift?
- **Larger dataset** — extend to Stanford Dogs (120 breeds) to test how the
  same architectures scale to a harder fine-grained problem.
- **Deployment scaling** — batched inference and ONNX export for CPU-only
  serving, plus a quantised int8 build of the winner to halve the artefact
  size again.

---

## Repository map

```
.
├── app.py                            # FastAPI app (GET /, POST /predict, GET /health)
├── templates/
│   └── index.html                    # upload page rendered by Jinja2
├── static/
│   ├── style.css                     # dark theme
│   ├── app.js                        # vanilla JS: fetch /predict + render top-5 bars
│   └── cats_and_dogs.jpg             # banner image served via /static
├── download_dataset.py               # one-shot dataset fetcher
├── cnn_model.keras                   # the winning model (overwritten by train_compare.py)
├── class_indices.json                # index → breed name, consumed by app.py
├── model_info.json                   # winner metadata, consumed by app.py
├── pyproject.toml                    # uv-managed deps
├── model_experiments/
│   ├── train_compare.py              # tournament entry point
│   ├── models/                       # one file per model factory
│   ├── results/                      # results_<date>.json + run_<date>.log
│   ├── saved_models/                 # *.keras per model + best_model.keras
│   └── plots/                        # learning curves, class distribution, speed-vs-accuracy plot
└── huggingface_space/                # Dockerfile + requirements.txt + Space README, copied at deploy time
```

## Deployment — Hugging Face Spaces (Docker SDK)

The Space is built from a Dockerfile that runs the FastAPI app under
`uvicorn` on port 7860. `huggingface_space/` contains the deploy manifest
(`Dockerfile`, `requirements.txt`, `README.md`); the rest of the Space's
contents — `app.py`, `templates/`, `static/`, and the three model
artefacts — are copied in from the repo root at deploy time.

To publish:

1. Create a new Space on Hugging Face with **SDK: Docker**.
2. `git clone` the Space, then from this repo run the cp block documented
   in `huggingface_space/DEPLOY.md`.
3. `git push` — HF builds the Docker image and starts the container
   automatically.

`cnn_model.keras` is typically larger than 10 MB, so the Space needs Git LFS
on first push (one-time setup steps are in `DEPLOY.md`).
