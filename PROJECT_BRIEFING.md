# Project Briefing — ML Model Tournament & Modern App Deployment
## For: Claude Opus (new session)
## Context: This document captures every decision, architectural choice, and confirmed specification from an extensive planning session. Build exactly what is described here — do not deviate unless explicitly noted as an open question.

---

## 1. Project Overview

This is an image classification project being modernized and expanded from a basic Keras CNN binary classifier (cats vs dogs) into a full ML model tournament with a deployable, explainable AI web application.

**Current state of the repo:**
- `app.py` — Streamlit app, loads `cnn_model.h5`, preprocesses uploaded image, predicts CAT or DOG
- `CNN.ipynb` — training notebook, builds a Sequential CNN, trains 25 epochs, saves `cnn_model.h5`
- `dataset_CNN/` — original dataset (cats vs dogs), confirmed perfectly balanced (4001/4001 train, 1001/1001 test), zero filename overlap between train and test
- `pyproject.toml` — managed by `uv`, tensorflow==2.15.0, keras==2.15.0, streamlit==1.29.0, numpy==1.26.2, pillow==10.1.0
- `CLAUDE.md` — project instructions, critical invariant: preprocessing must match between notebook and app
- `cats_and_dogs.jpg` — banner image for the app

**What we are building:**
1. A `model_experiments/` folder containing a full model tournament script
2. An upgraded `app.py` with modern UI, Grad-CAM, confidence bar chart, model card
3. A comprehensive `README.md` structured as a research paper
4. Deployment to Hugging Face Spaces

---

## 2. Dataset Decision

**Switch from Kaggle Cats vs Dogs → Oxford-IIIT Pet Dataset**

- 37 classes (cat and dog breeds)
- ~200 images per breed, ~7,400 total images
- Free download from Oxford: https://www.robots.ox.ac.uk/~vgg/data/pets/
- This changes the problem from binary classification to 37-class classification
- Output layer: `Dense(37, softmax)` instead of `Dense(1, sigmoid)`
- Loss function: `categorical_crossentropy` instead of `binary_crossentropy`
- The dataset needs to be downloaded as part of setup — handle this in a `download_dataset.py` script or at the top of `train_compare.py`
- Class encoding: Keras assigns classes alphabetically by breed folder name

**Why Oxford-IIIT Pet:**
- Fine-grained visual classification (distinguishing breeds) is genuinely non-trivial
- Direct natural evolution of the cats vs dogs project
- Has real business applications (pet insurance, veterinary triage, adoption platforms)
- Results are actually meaningful — not a solved problem like binary cats vs dogs

### Class Imbalance Analysis — Required Step
Oxford-IIIT Pet is approximately balanced (~200 images per class) but NOT perfectly balanced. Before training, the tournament script must count images per class and log the distribution.

**In `train_compare.py`:** After loading the dataset, count images per class, log min/max/mean counts, and flag any class with fewer than 150 images.

**In README.md:** Include a horizontal bar chart (matplotlib) showing image counts per breed, sorted ascending. This makes the dataset honest and transparent — a recruiter or reviewer will appreciate seeing you checked this rather than assuming balance.

**Metric implication:** Because classes are not perfectly balanced, **macro F1** is the primary metric for evaluating classification quality — it weights all classes equally regardless of size. Accuracy alone will be inflated by the more common classes. This is already captured in the metrics plan (Section 6) but the reasoning should be stated in the README Methodology section.

---

## 3. Model Tournament — Full Specification

### Folder Structure
```
model_experiments/
├── train_compare.py           # single entry point — run this
├── models/
│   ├── __init__.py            # empty, marks as Python package
│   ├── baseline_cnn.py        # returns compiled Keras model
│   ├── deeper_cnn.py
│   ├── mobilenet_frozen.py
│   ├── mobilenet_finetune.py
│   ├── efficientnetb0_frozen.py
│   └── efficientnetb0_finetune.py
├── results/
│   └── results_YYYY-MM-DD.json   # auto-named by run date
├── saved_models/
│   ├── baseline_cnn.h5
│   ├── deeper_cnn.h5
│   ├── mobilenet_frozen.h5
│   ├── mobilenet_finetune.h5
│   ├── efficientnetb0_frozen.h5
│   ├── efficientnetb0_finetune.h5
│   └── best_model.h5             # winner copied here AND to ../cnn_model.h5
└── plots/
    └── learning_curves_YYYY-MM-DD/
        ├── baseline_cnn.png
        ├── deeper_cnn.png
        └── ... (one plot per model)
```

### Model Lineup (6 models — ResNet50 was explicitly dropped)
**Note on ResNet50:** It was considered but dropped due to ~100MB model size with no meaningful accuracy advantage over EfficientNetB0. It MUST be mentioned in the README under Future Work or Methodology with this reasoning.

| # | Model | Type | Preprocessing | Epochs |
|---|---|---|---|---|
| 1 | Baseline CNN | Custom from scratch | `/255.0` | 25 |
| 2 | Deeper CNN | Custom from scratch | `/255.0` | 25 |
| 3 | MobileNetV2 (frozen) | Pretrained, head only | `mobilenet_v2.preprocess_input` | 15 |
| 4 | MobileNetV2 (fine-tuned) | Pretrained, top 20% unfrozen | `mobilenet_v2.preprocess_input` | 10 |
| 5 | EfficientNetB0 (frozen) | Pretrained, head only | `efficientnet.preprocess_input` | 15 |
| 6 | EfficientNetB0 (fine-tuned) | Pretrained, top 20% unfrozen | `efficientnet.preprocess_input` | 10 |

### Model Architectures

**Model 1 — Baseline CNN:**
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(37, activation='softmax', dtype='float32')  # float32 explicit for mixed precision
])
# Compiled: adam (lr=1e-3), categorical_crossentropy, metrics=['accuracy']
```

**Model 2 — Deeper CNN:**
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(37, activation='softmax', dtype='float32')
])
# Compiled: adam (lr=1e-3), categorical_crossentropy, metrics=['accuracy']
```

**Model 3 — MobileNetV2 (frozen):**
```python
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(37, activation='softmax', dtype='float32')(x)
model = Model(base.input, output)
# Compiled: adam (lr=1e-3), categorical_crossentropy, metrics=['accuracy']
```

**Model 4 — MobileNetV2 (fine-tuned):**
- Start from Model 3's trained weights (train frozen first, then unfreeze)
- Unfreeze top 20% of MobileNetV2 layers (MobileNetV2 has 155 layers → unfreeze last ~31)
- Recompile with lr=1e-5 (CRITICAL — must be much lower than head training lr)
- Train for 10 more epochs with early stopping

**Model 5 — EfficientNetB0 (frozen):**
```python
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(37, activation='softmax', dtype='float32')(x)
model = Model(base.input, output)
# Compiled: adam (lr=1e-3), categorical_crossentropy, metrics=['accuracy']
```

**Model 6 — EfficientNetB0 (fine-tuned):**
- Start from Model 5's trained weights
- Unfreeze top 20% of EfficientNetB0 layers (237 layers → unfreeze last ~47)
- Recompile with lr=1e-5
- Train for 10 more epochs with early stopping

---

## 4. Training Configuration — All Confirmed

### Input
- **Input size: 224×224×3** — confirmed, gives pretrained models their full designed resolution
- **Batch size: 32** — confirmed, optimal for M1 16GB with tensorflow-metal

### Validation Split
- **20% of training set carved out as validation**
- Done in-script — NO new folders created on disk, folder structure unchanged
- Use fixed random seed when splitting so it's reproducible
- The remaining 80% is training data
- `test_set/` is NEVER touched during training — used exactly once at the very end for final evaluation

### Early Stopping
- **Patience: 5** — same value across ALL 6 models, no exceptions
- Monitor: `val_loss`
- `restore_best_weights=True` — always restore to best checkpoint, not last epoch

### Learning Rates
- Head training (all frozen models): `lr = 1e-3`
- Fine-tuning phase: `lr = 1e-5` — CRITICAL, must be this low to avoid destroying pretrained weights

### Preprocessing — Model Specific (Option A — confirmed)
Each model gets its own correct preprocessing function:
```python
# Custom CNNs
image / 255.0

# MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
preprocess_input(image)  # scales to [-1, 1]

# EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
preprocess_input(image)  # scales to [-1, 1]
```
This means each model performs at its genuine best. The comparison is intentionally not apples-to-apples on input values — that is acceptable because the goal is finding the best model for deployment, not a controlled lab comparison.

### Random Seed — Fixed
```python
import random, numpy as np, tensorflow as tf
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
Set this at the very top of `train_compare.py` before any imports that use randomness.

---

## 5. Three Runtime Optimizations — All Must Be Implemented

### Optimization 1 — Mixed Precision Training
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```
- Set at the very top of `train_compare.py` before model definitions
- CRITICAL: All output layers must explicitly use `dtype='float32'` to avoid numerical instability
- This is already reflected in all model architectures above

### Optimization 2 — Parallel Data Loading
**CRITICAL:** Do NOT use `workers=4, use_multiprocessing=True` in `model.fit()` when passing a `tf.data.Dataset`. Those parameters belong to the Keras `Sequence` / generator API — passing them with a `tf.data.Dataset` either errors or is silently ignored. They are mutually exclusive APIs.

The correct approach when using `tf.data.Dataset` is to let tf.data handle all parallelism internally:
```python
train_dataset = (
    tf.data.Dataset.from_tensor_slices(...)
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # parallel CPU decode
    .cache()
    .shuffle(buffer_size)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)  # overlap CPU prep with GPU training
)
```
- `num_parallel_calls=tf.data.AUTOTUNE` → TensorFlow picks the optimal thread count for the machine
- `.prefetch(tf.data.AUTOTUNE)` → CPU prepares next batch while GPU trains on current batch
- Do NOT pass `workers` or `use_multiprocessing` to `model.fit()` — omit those arguments entirely

### Optimization 3 — Dataset Caching in Memory
- Convert `ImageDataGenerator` flows to `tf.data.Dataset` with `.cache()`
- M1 16GB has enough RAM to cache the entire Oxford-IIIT Pet dataset
- Disk reads only happen on epoch 1; all subsequent epochs read from RAM
```python
train_dataset = tf.data.Dataset.from_generator(...).cache().shuffle(buffer_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_generator(...).cache().batch(32).prefetch(tf.data.AUTOTUNE)
```

### Additional: tensorflow-metal
- Add `tensorflow-metal` to `pyproject.toml` dependencies
- This enables M1 GPU acceleration via Apple's Metal framework
- Without it, TensorFlow only uses CPU on Mac

---

## 6. Metrics & Logging — All Confirmed

### Metrics to log per model
- `train_accuracy` (final epoch)
- `val_accuracy` (best epoch)
- `test_accuracy` (run once at end on test_set/)
- `train_loss`
- `val_loss`
- `precision` (per class + macro average)
- `recall` (per class + macro average)
- `f1_score` (per class + macro average)
- `params_total` — total trainable parameters
- `training_time_seconds`
- `avg_inference_time_ms` — average over 100 test images (inference speed, not training speed)
- `model_size_mb`
- `epochs_trained` — how many epochs before early stopping fired
- `best_epoch` — which epoch had best val_loss
- `early_stopped` — boolean
- `saved_to` — path to saved .h5 file
- `training_history` — full epoch-by-epoch loss and accuracy arrays for plotting

### JSON Schema
```json
{
  "run_date": "2026-05-19",
  "seed": 42,
  "dataset": {
    "name": "Oxford-IIIT Pet",
    "num_classes": 37,
    "train_samples": 5920,
    "val_samples": 1480,
    "test_samples": 3669,
    "input_size": [224, 224],
    "batch_size": 32
  },
  "models": [
    {
      "name": "baseline_cnn",
      "params_total": 0,
      "epochs_trained": 0,
      "best_epoch": 0,
      "train_accuracy": 0.0,
      "val_accuracy": 0.0,
      "test_accuracy": 0.0,
      "train_loss": 0.0,
      "val_loss": 0.0,
      "precision_macro": 0.0,
      "recall_macro": 0.0,
      "f1_macro": 0.0,
      "training_time_seconds": 0.0,
      "avg_inference_time_ms": 0.0,
      "model_size_mb": 0.0,
      "early_stopped": true,
      "saved_to": "saved_models/baseline_cnn.h5",
      "training_history": {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": []
      },
      "status": "completed"
    }
  ],
  "winner": "efficientnetb0_finetune",
  "winner_test_accuracy": 0.0,
  "speed_accuracy_tradeoff": [
    {
      "model": "baseline_cnn",
      "test_accuracy": 0.0,
      "avg_inference_time_ms": 0.0
    }
  ]
}
```

### Silent Failure Handling — Confirmed Approach
- Wrap each model's entire training block in `try/except`
- Write results to JSON immediately after each model completes — do NOT wait until all models finish
- On failure: log the error, write `"status": "failed", "error": str(e)` to that model's JSON entry, continue to next model
- The script must never halt entirely due to a single model failure

```python
for model_config in model_configs:
    try:
        # train, evaluate, log
        results["models"].append(model_results)
    except Exception as e:
        logger.error(f"Model {model_config['name']} failed: {e}")
        results["models"].append({"name": model_config["name"], "status": "failed", "error": str(e)})
    finally:
        # always write JSON after each model, success or failure
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
```

### Logging
- Use Python `logging` module
- Log to both console (with tqdm-compatible handler) and `results/run_YYYY-MM-DD.log`
- tqdm progress bars per epoch via custom Keras callback wrapping tqdm

### Learning Curve Plots
- After each model, save a matplotlib plot of train vs val accuracy and loss across epochs
- Save to `plots/learning_curves_YYYY-MM-DD/model_name.png`
- These are used in the README

### Winner Selection
- Winner is selected by **val_accuracy** — NOT test_accuracy
- This preserves the integrity of the test set: it is touched exactly once, only for the final winner's performance report
- After the winner is selected by val_accuracy, run the test set through the winner model only — report this as the final "real-world" performance figure
- Winner's .h5 is copied to `saved_models/best_model.h5` AND to `../cnn_model.h5` so app.py uses it immediately
- **Why this matters:** selecting the winner by test accuracy turns the test set into a selection criterion, which makes it validation by another name. The test set must never influence any training or model selection decision.

---

## 7. Speed vs Accuracy Tradeoff Analysis

- Time **inference**, not training — training time is irrelevant in production
- Run **500 predictions** on test images per model (minimum), measure total time, divide by count
- Use **median**, not mean — M1 GPU (Metal) has warmup latency on early predictions that will skew the mean upward. Median gives a stable, representative figure.
- Log `avg_inference_time_ms` to JSON for each model
- This forms a scatter plot of accuracy vs inference speed — 6 data points, one per model
- Ideal model: top-left corner (high accuracy, fast inference)
- This analysis is included in the README results section
- **Note on terminology:** this is technically a scatter plot; "Pareto frontier" is used loosely in the README to describe the accuracy/speed tradeoff concept, not a formal non-dominated set calculation

---

## 8. app.py — Modern UI Redesign

### Stack
- Streamlit (existing)
- Plotly for the confidence bar chart (interactive)
- `tf-keras-vis` for Grad-CAM heatmap overlay
- PIL / OpenCV for heatmap blending

### UI Flow
```
1. User uploads image
2. Image displayed as styled card
3. Model preprocesses image (224×224, model-specific preprocessing)
4. Prediction runs → softmax output over 37 classes
5. Top-5 breeds shown as horizontal bar chart (Plotly)
   - X axis: confidence percentage
   - Y axis: breed names
   - Style: dark theme, gradient bars (reference image provided — horizontal bars with values on right)
6. Grad-CAM heatmap generated and overlaid on original image
   - Jet colormap, ~60% opacity
   - Original image | Heatmap overlay shown side by side
7. Model card shown bottom-right corner
   - Architecture name
   - Test accuracy
   - Total parameters
   - Brief description
```

### Grad-CAM Implementation
- Use `tf-keras-vis` library — confirmed, do NOT implement manually
- Target layer differs per model architecture:
  - Baseline CNN / Deeper CNN → last `Conv2D` layer name
  - MobileNetV2 → `out_relu`
  - EfficientNetB0 → `top_activation`
- Store a config dictionary in app.py mapping model name → target layer name
- The app must know which model is loaded to target the correct layer

### Class Change Impact on app.py
- No longer outputs CAT/DOG — outputs breed name + confidence
- `cnn_model.h5` now outputs 37 classes (softmax)
- Preprocessing in app.py must match the winner model's preprocessing function
- app.py needs to know the class index → breed name mapping (save this as `class_indices.json` during training)

---

## 9. README Structure — Confirmed

Order confirmed as:
```
1. Introduction
2. Business Context
3. Hypothesis
4. Methodology
5. Results
6. Conclusions
7. Future Work
```

### Content per section:

**Introduction:** What this project is, what problem it solves, what the tournament approach demonstrates.

**Business Context:** Breed identification has real applications — pet insurance (automated breed verification), veterinary triage (breed-specific health risks), adoption platforms (automated breed tagging from photos). The explainability layer (Grad-CAM) addresses the regulatory and trust requirement for AI decisions to be interpretable.

**Hypothesis:** 
- "Transfer learning models (MobileNetV2, EfficientNetB0) will significantly outperform custom CNNs on the Oxford-IIIT Pet dataset because our dataset size (~7,400 images across 37 classes) is too small for deep networks to learn robust fine-grained features from scratch."
- "Fine-tuning the top 20% of EfficientNetB0 will yield the highest overall accuracy, but the marginal gain over the frozen variant may be modest — pretrained ImageNet features already encode strong general visual representations."
- "There will be a meaningful speed vs accuracy tradeoff: custom CNNs will have faster inference times but lower accuracy; EfficientNetB0 fine-tuned will have the highest accuracy but slowest inference."

**Methodology:** Dataset, all 6 models with reasoning, training setup (all hyperparameters), the three optimizations, validation strategy, evaluation metrics.

**Note on ResNet50:** Must be mentioned here — it was considered, dropped because at ~100MB with no meaningful accuracy advantage over EfficientNetB0 (~20MB), it was not cost-effective. This shows pragmatic engineering judgment.

**Results:** Metrics table for all 6 models, learning curve plots, speed vs accuracy Pareto plot.

**Conclusions:** What the data showed, was hypothesis confirmed, which model won and why.

**Future Work:** 
- ResNet50 inclusion for completeness
- Failure analysis (clustering misclassified images by brightness, size, class)
- Input size upgrade to 224×224 fine-tuning experiments
- Larger dataset (Stanford Dogs for 120-class fine-grained classification)
- Deployment scaling considerations

---

## 10. Deployment — Hugging Face Spaces

- Platform: **Hugging Face Spaces** — confirmed, no objection raised
- Framework: Streamlit (already used)
- The Space should have a proper model card
- `best_model.h5` + `class_indices.json` uploaded to the Space
- README on the Space page links back to the GitHub repo

---

## 11. Data Augmentation Strategy

Apply to training data only — NEVER to validation or test data.

Confirmed augmentations (domain-reasoned for pet photos):
- **Horizontal flip** ✅ — flipped cat/dog is still a cat/dog
- **Random rotation (±15°)** ✅ — animals aren't always perfectly upright
- **Brightness adjustment (±20%)** ✅ — handles different lighting conditions
- **Zoom (up to 10%)** ✅ — animals at different distances, conservative value
- **Vertical flip** ❌ — upside down animals are unrealistic, excluded
- **Aggressive shear/distortion** ❌ — destroys recognizable features, excluded

---

## 12. Dependencies to Add to pyproject.toml

```toml
dependencies = [
    "streamlit==1.29.0",
    "tensorflow==2.15.0",
    "keras==2.15.0",
    "numpy==1.26.2",
    "pillow==10.1.0",
    "tensorflow-metal",       # M1 GPU acceleration — ADD THIS
    "tf-keras-vis",           # Grad-CAM — ADD THIS
    "plotly",                 # Interactive confidence bar chart — ADD THIS
    "scikit-learn",           # F1, precision, recall metrics — ADD THIS
    "tqdm",                   # Progress bars — ADD THIS
    "matplotlib",             # Learning curve plots — ADD THIS
    "opencv-python",          # Heatmap blending for Grad-CAM — ADD THIS
]
```

---

## 13. System Information

- **Machine:** Apple M1, 16GB RAM
- **OS:** macOS
- **Package manager:** `uv`
- **Python:** >=3.9, <3.12 (tensorflow 2.15.0 constraint)
- **Run command:** `uv run python model_experiments/train_compare.py`

---

## 14. What Has NOT Been Decided / Open Items

- Whether to implement `download_dataset.py` as a separate script or inline at the top of `train_compare.py`
- Exact Streamlit UI color scheme / theme (dark theme implied but not specified)
- Whether `plot_results.py` is a separate script or plots are generated inline during training
- Hugging Face Space name and organization

---

## 15. Build Order Recommendation

1. Add dependencies to `pyproject.toml`, run `uv sync`
2. Download Oxford-IIIT Pet dataset, set up folder structure
3. Build `model_experiments/models/` — all 6 model definition files + `__init__.py`
4. Build `train_compare.py` — full tournament script with all logging, optimizations, JSON output
5. Run the tournament, verify JSON output is correct
6. Build upgraded `app.py` — modern UI, Grad-CAM, confidence bar chart, model card
7. Write `README.md`
8. Deploy to Hugging Face Spaces

---

*This briefing document was generated from an extensive planning session. Every item marked "confirmed" was explicitly agreed upon. Items in Section 14 are genuinely open and should be decided before or during implementation.*
