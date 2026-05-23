"""Six-model tournament on Oxford-IIIT Pet.

Single entry point. Trains each model, evaluates it on the validation set
(per-class P/R/F1 + macros), and writes incremental results JSON. Winner is
chosen by `val_accuracy` and is the *only* model evaluated on the held-out
test set — that final test pass is the one "real-world" performance figure.
The winning .h5 is then copied to `../cnn_model.h5` so `app.py` picks it up
immediately.

Run with:

    uv run python model_experiments/train_compare.py

Prerequisites:
    1. `uv sync`
    2. `uv run python download_dataset.py`

Design notes
------------
* Mixed precision is enabled globally; every model's output `Dense` is
  pinned to float32 (in its builder file) for numerical stability.
* Images are loaded once, resized to 224x224, and cached in RAM as uint8.
  Model-specific preprocessing (and training-time augmentation) is then
  applied via a cheap `.map` step that runs from cache each epoch.
* Per-model training is wrapped in `try/except`; the JSON file is rewritten
  after every model so a mid-run crash never throws away earlier results.
* Fine-tune models (mobilenet_finetune, efficientnetb0_finetune) consume the
  in-memory weights from their frozen counterpart — the frozen `.keras` on
  disk was already saved before mutation.
* Model artefacts are saved in Keras 3's native `.keras` format (zip-archive).
  The `.h5` legacy path is deprecated in Keras 3 and emits noisy warnings.
"""

from __future__ import annotations

# ── 0. Make `from model_experiments.* import ...` work when this script is ──
# invoked directly (python model_experiments/train_compare.py) rather than as
# a module (python -m model_experiments.train_compare). Direct invocation puts
# only `model_experiments/` on sys.path; we also need the repo root.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# ── 1. Seed everything BEFORE any randomness-using import ────────────────
import os
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np  # noqa: E402

np.random.seed(SEED)

import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)

# ── 2. Mixed precision BEFORE any model is built ────────────────────────
from keras import mixed_precision  # noqa: E402

mixed_precision.set_global_policy("mixed_float16")

# ── 3. Remaining imports ────────────────────────────────────────────────
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from math import prod  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import cast  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from model_experiments.models import (  # noqa: E402
    baseline_cnn,
    deeper_cnn,
    efficientnetb0_finetune,
    efficientnetb0_frozen,
    mobilenet_finetune,
    mobilenet_frozen,
)

# ── 4. Constants ────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_FRACTION = 0.20
EARLY_STOP_PATIENCE = 5
INFERENCE_BENCHMARK_N = 500
NUM_CLASSES = 37
SHUFFLE_BUFFER = 2048

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset_oxford_pets"
OUT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = OUT_ROOT / "saved_models"
RESULTS_DIR = OUT_ROOT / "results"

logger = logging.getLogger("tournament")


# ── 5. Dataset discovery & split ────────────────────────────────────────
def discover(root: Path, split: str) -> tuple[list[str], list[int], list[str]]:
    """Return (image_paths, labels, class_names) for `root/split/<class>/*.jpg`."""
    base = root / split
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist; run download_dataset.py first")
    class_names = sorted([d.name for d in base.iterdir() if d.is_dir()])
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    for cls in class_names:
        for img in sorted((base / cls).iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(img))
                labels.append(name_to_idx[cls])
    return paths, labels, class_names


def stratified_split(
    paths: list[str], labels: list[int], val_frac: float, seed: int
) -> tuple[list[str], list[int], list[str], list[int]]:
    """80 / 20 stratified split with fixed seed — class proportions preserved."""
    rng = np.random.default_rng(seed)
    paths_arr = np.array(paths)
    labels_arr = np.array(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels_arr):
        idx = np.where(labels_arr == cls)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_frac))
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        paths_arr[train_idx].tolist(),
        labels_arr[train_idx].tolist(),
        paths_arr[val_idx].tolist(),
        labels_arr[val_idx].tolist(),
    )


# ── 6. tf.data pipeline ─────────────────────────────────────────────────
def _load_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.uint8)            # cache as uint8 to save RAM
    return image, label


def make_raw_dataset(paths: list[str], labels: list[int]) -> tf.data.Dataset:
    """Decode + resize once, then cache in RAM. Augmentation/preprocessing later."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache()


augment_layer = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(15.0 / 360.0, seed=SEED),  # ±15°, fraction-of-2π convention
        tf.keras.layers.RandomBrightness(0.20, seed=SEED),
        tf.keras.layers.RandomZoom(0.10, seed=SEED),
    ],
    name="augmentation",
)




def preprocess_fn_for(name: str):
    """Map a preprocessing tag (declared in each model file) to a callable."""
    if name == "rescale_1_over_255":
        return lambda x: x / 255.0
    if name == "mobilenet_v2":
        from keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    if name == "efficientnet":
        from keras.applications.efficientnet import preprocess_input
        return preprocess_input
    raise ValueError(f"Unknown preprocessing tag: {name}")


def make_dataset(raw_ds: tf.data.Dataset, preprocess_name: str, training: bool) -> tf.data.Dataset:
    """Layer training-only augmentation and model-specific preprocessing on top of the cached raw."""
    pp = preprocess_fn_for(preprocess_name)

    def transform(image, label):
        image = tf.cast(image, tf.float32)
        if training:
            image = augment_layer(image, training=True)
        image = pp(image)
        return image, tf.one_hot(label, NUM_CLASSES)

    ds = raw_ds
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── 7. tqdm callback (one bar per model, ticks per epoch) ───────────────
class TqdmEpochBar(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, label: str) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.label = label
        self.bar: tqdm | None = None

    def on_train_begin(self, logs=None):
        self.bar = tqdm(total=self.total_epochs, desc=self.label, unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        if self.bar and logs:
            self.bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
            self.bar.update(1)

    def on_train_end(self, logs=None):
        if self.bar:
            self.bar.close()


# ── 8. Inference timing — 500 single-image predictions, GPU-warmed ──────
def time_inference(model: tf.keras.Model, raw_images: tf.data.Dataset, preprocess_name: str) -> float:
    """Median per-image prediction latency in ms.

    Median (not mean) because the Metal backend has warm-up latency on the
    first handful of predictions that drags the mean upward; median is the
    stable representative figure.
    """
    pp = preprocess_fn_for(preprocess_name)
    samples = list(raw_images.take(INFERENCE_BENCHMARK_N).as_numpy_iterator())
    if len(samples) < INFERENCE_BENCHMARK_N:
        return float("nan")
    preprocessed = [pp(tf.cast(img, tf.float32)).numpy() for img, _ in samples]

    # Warm-up (kernel compilation, weight upload to Metal)
    _ = model.predict(np.expand_dims(preprocessed[0], 0), verbose=0)

    timings: list[float] = []
    for x in preprocessed:
        t0 = time.perf_counter()
        _ = model.predict(np.expand_dims(x, 0), verbose=0)
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings)) * 1000.0


# ── 9. Plots ────────────────────────────────────────────────────────────
def log_class_counts(class_names: list[str], labels: list[int]) -> np.ndarray:
    """Log min/max/mean image counts; warn on any class < 150 images."""
    counts = np.bincount(labels, minlength=len(class_names))
    logger.info(
        f"Class distribution (trainval): min={int(counts.min())} "
        f"max={int(counts.max())} mean={counts.mean():.1f}"
    )
    flagged = [class_names[i] for i, c in enumerate(counts) if c < 150]
    if flagged:
        logger.warning(f"Classes with <150 images: {flagged}")
    return counts


def plot_class_distribution(class_names: list[str], counts: np.ndarray, path: Path) -> None:
    """Horizontal bar chart of per-class image counts, sorted ascending."""
    order = np.argsort(counts)
    sorted_names = [class_names[i] for i in order]
    sorted_counts = counts[order]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(range(len(sorted_names)), sorted_counts, color="#7c5cff")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.axvline(150, color="red", linestyle="--", alpha=0.6, label="150-image floor")
    ax.set(xlabel="image count (trainval)", title="Oxford-IIIT Pet — class distribution")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def plot_learning_curve(history: tf.keras.callbacks.History, path: Path, name: str) -> None:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, h["accuracy"], label="train")
    axes[0].plot(epochs, h["val_accuracy"], label="val")
    axes[0].set(xlabel="epoch", ylabel="accuracy", title=f"{name} — accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, h["loss"], label="train")
    axes[1].plot(epochs, h["val_loss"], label="val")
    axes[1].set(xlabel="epoch", ylabel="loss", title=f"{name} — loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def plot_speed_accuracy(results: dict, path: Path) -> None:
    """Scatter of median inference latency vs validation accuracy, one point per model.

    Y-axis is val_accuracy (not test) because the test set is touched only
    once, on the tournament winner — only val_accuracy is comparable across
    all 6 models.
    """
    points = [
        (m["avg_inference_time_ms"], m["val_accuracy"], m["name"])
        for m in results["models"]
        if m.get("status") == "completed"
    ]
    if not points:
        return
    xs, ys, _ = zip(*points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=90, color="#7c5cff")
    for x, y, n in points:
        ax.annotate(n, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set(
        xlabel="Median inference time per image (ms)",
        ylabel="Validation accuracy",
        title="Speed vs Accuracy",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


# ── 10. Train one model ─────────────────────────────────────────────────
def _trainable_param_count(model: tf.keras.Model) -> int:
    # Keras 3 Variable.shape is typed as tuple[int | None, ...] — `None` means a
    # dynamic dim, which never happens for *weights* (only for input tensors), so
    # filter out the None branch to satisfy the type checker.
    return sum(
        prod(int(d) for d in v.shape if d is not None)
        for v in model.trainable_weights
    )


def train_one_model(
    cfg: dict,
    raw_train: tf.data.Dataset,
    raw_val: tf.data.Dataset,
    raw_test: tf.data.Dataset,
    class_names: list[str],
    results: dict,
    results_path: Path,
    plots_dir: Path,
    in_memory: dict[str, tf.keras.Model],
) -> dict:
    name = cfg["name"]
    logger.info("=" * 64)
    logger.info(f"  TRAIN  ▶  {name}  ({cfg['epochs']} epochs max)")
    logger.info("=" * 64)
    result: dict = {"name": name}

    try:
        train_ds = make_dataset(raw_train, cfg["preprocessing"], training=True)
        val_ds = make_dataset(raw_val, cfg["preprocessing"], training=False)

        if cfg["depends_on"]:
            parent = in_memory.get(cfg["depends_on"])
            if parent is None:
                raise RuntimeError(f"{name} depends on {cfg['depends_on']} which is unavailable")
            model = cfg["builder"](parent)
        else:
            model = cfg["builder"](num_classes=NUM_CLASSES)

        params_trainable = _trainable_param_count(model)
        logger.info(f"  trainable params: {params_trainable:,}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            TqdmEpochBar(cfg["epochs"], label=name),
        ]

        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["epochs"],
            callbacks=callbacks,
            verbose=0,
        )
        training_time = time.time() - t0

        # Per-class P/R/F1 on the validation set. The test set is held back
        # entirely; it is touched once, only on the tournament winner, after
        # winner selection in main().
        y_true: list[int] = []
        y_pred: list[int] = []
        for x, y in val_ds:
            y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
            y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1).tolist())
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        # average=None returns four ndarrays, but sklearn's stubs union with the
        # scalar form (used when average is a string). Cast to narrow.
        prec_pc, rec_pc, f1_pc, support_pc = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
            precision_recall_fscore_support(
                y_true_arr, y_pred_arr, average=None, zero_division=0,
                labels=list(range(NUM_CLASSES)),
            ),
        )

        # Inference latency (timing only — no labels read, no selection signal)
        avg_ms = time_inference(model, raw_test, cfg["preprocessing"])

        # Persist weights (Keras 3 native zip-archive format)
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = SAVED_MODELS_DIR / f"{name}.keras"
        model.save(model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)

        # Learning curve plot
        plot_learning_curve(history, plots_dir / f"{name}.png", name)

        result.update(
            {
                "params_total": params_trainable,
                "epochs_trained": len(history.epoch),
                "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                "train_accuracy": float(history.history["accuracy"][-1]),
                "val_accuracy": float(max(history.history["val_accuracy"])),
                "train_loss": float(history.history["loss"][-1]),
                "val_loss": float(min(history.history["val_loss"])),
                "precision_macro": float(p_macro),
                "recall_macro": float(r_macro),
                "f1_macro": float(f1_macro),
                "per_class": [
                    {
                        "class": class_names[i],
                        "precision": float(prec_pc[i]),
                        "recall": float(rec_pc[i]),
                        "f1": float(f1_pc[i]),
                        "support": int(support_pc[i]),
                    }
                    for i in range(NUM_CLASSES)
                ],
                "training_time_seconds": float(training_time),
                "avg_inference_time_ms": float(avg_ms),
                "model_size_mb": float(size_mb),
                "early_stopped": len(history.epoch) < cfg["epochs"],
                "saved_to": str(model_path.relative_to(REPO_ROOT)),
                "training_history": {
                    "train_accuracy": [float(x) for x in history.history["accuracy"]],
                    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                    "train_loss": [float(x) for x in history.history["loss"]],
                    "val_loss": [float(x) for x in history.history["val_loss"]],
                },
                "preprocessing": cfg["preprocessing"],
                "gradcam_target_layer": cfg["gradcam_target"],
                "status": "completed",
            }
        )

        # Save (frozen) model so its fine-tune sibling can pick up the trained weights
        in_memory[name] = model
        logger.info(
            f"  ✓ {name}  val_acc={float(max(history.history['val_accuracy'])):.4f}  "
            f"f1_macro={f1_macro:.4f}  size={size_mb:.1f}MB  inf={avg_ms:.1f}ms"
        )
    except Exception as e:
        logger.exception(f"Model {name} failed")
        result.update({"status": "failed", "error": str(e)})
    finally:
        existing = next((i for i, m in enumerate(results["models"]) if m["name"] == name), None)
        if existing is not None:
            results["models"][existing] = result
        else:
            results["models"].append(result)
        results_path.write_text(json.dumps(results, indent=2))

    return result


# ── 11. Main ────────────────────────────────────────────────────────────
def main() -> int:
    today = datetime.date.today().isoformat()
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_ROOT / "plots" / f"learning_curves_{today}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"run_{today}.log"
    results_path = RESULTS_DIR / f"results_{today}.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    logger.info(f"GPUs visible to TF: {tf.config.list_physical_devices('GPU')}")
    logger.info(f"Mixed precision policy: {mixed_precision.global_policy().name}")

    if not DATASET_ROOT.exists():
        logger.error(f"Dataset not found at {DATASET_ROOT}. Run download_dataset.py first.")
        return 1

    logger.info("Discovering dataset…")
    trainval_paths, trainval_labels, class_names = discover(DATASET_ROOT, "trainval")
    test_paths, test_labels, test_class_names = discover(DATASET_ROOT, "test")
    if class_names != test_class_names:
        raise RuntimeError("Class lists differ between trainval/ and test/")
    if len(class_names) != NUM_CLASSES:
        logger.warning(f"Expected {NUM_CLASSES} classes, found {len(class_names)}")

    # Class-imbalance honesty check (Briefing §2): the dataset is approximately
    # but not perfectly balanced; the bar chart goes into the README.
    counts = log_class_counts(class_names, trainval_labels)
    plot_class_distribution(
        class_names, counts, OUT_ROOT / "plots" / "class_distribution.png"
    )

    train_p, train_l, val_p, val_l = stratified_split(
        trainval_paths, trainval_labels, VAL_FRACTION, SEED
    )
    logger.info(f"  train={len(train_p)}  val={len(val_p)}  test={len(test_paths)}")

    # class_indices.json is consumed by app.py: index → breed name
    class_indices = {str(i): name for i, name in enumerate(class_names)}
    (REPO_ROOT / "class_indices.json").write_text(json.dumps(class_indices, indent=2))

    raw_train = make_raw_dataset(train_p, train_l)
    raw_val = make_raw_dataset(val_p, val_l)
    raw_test = make_raw_dataset(test_paths, test_labels)

    results: dict = {
        "run_date": today,
        "seed": SEED,
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "num_classes": len(class_names),
            "train_samples": len(train_p),
            "val_samples": len(val_p),
            "test_samples": len(test_paths),
            "input_size": list(IMAGE_SIZE),
            "batch_size": BATCH_SIZE,
            "class_names": class_names,
            "class_counts_trainval": {
                class_names[i]: int(counts[i]) for i in range(len(class_names))
            },
        },
        "models": [],
        "winner": None,
        "winner_selection_metric": "val_accuracy",
        "winner_test_accuracy": None,
        "winner_test_f1_macro": None,
        "winner_test_precision_macro": None,
        "winner_test_recall_macro": None,
        "winner_test_per_class": [],
        "speed_accuracy_tradeoff": [],
    }
    results_path.write_text(json.dumps(results, indent=2))

    model_configs = [
        {"module": baseline_cnn, "epochs": 25, "depends_on": None},
        {"module": deeper_cnn, "epochs": 25, "depends_on": None},
        {"module": mobilenet_frozen, "epochs": 15, "depends_on": None},
        {"module": mobilenet_finetune, "epochs": 10, "depends_on": "mobilenet_frozen"},
        {"module": efficientnetb0_frozen, "epochs": 15, "depends_on": None},
        {"module": efficientnetb0_finetune, "epochs": 10, "depends_on": "efficientnetb0_frozen"},
    ]
    for cfg in model_configs:
        m = cfg["module"]
        cfg["name"] = m.NAME
        cfg["builder"] = m.build
        cfg["preprocessing"] = m.PREPROCESSING
        cfg["gradcam_target"] = m.GRADCAM_TARGET_LAYER

    in_memory: dict[str, tf.keras.Model] = {}
    for cfg in model_configs:
        train_one_model(
            cfg, raw_train, raw_val, raw_test, class_names,
            results, results_path, plots_dir, in_memory,
        )

    # Speed-vs-accuracy snapshot uses val_accuracy because only the eventual
    # winner is evaluated on the test set (Briefing §6).
    results["speed_accuracy_tradeoff"] = [
        {
            "model": m["name"],
            "val_accuracy": m["val_accuracy"],
            "avg_inference_time_ms": m["avg_inference_time_ms"],
        }
        for m in results["models"]
        if m.get("status") == "completed"
    ]

    completed = [m for m in results["models"] if m.get("status") == "completed"]
    if completed:
        winner = max(completed, key=lambda m: m["val_accuracy"])
        results["winner"] = winner["name"]
        logger.info(
            f"🏆 winner (by val_accuracy): {winner['name']}  "
            f"val_accuracy={winner['val_accuracy']:.4f}"
        )

        winner_artefact = SAVED_MODELS_DIR / f"{winner['name']}.keras"
        shutil.copy2(winner_artefact, SAVED_MODELS_DIR / "best_model.keras")
        shutil.copy2(winner_artefact, REPO_ROOT / "cnn_model.keras")  # app.py picks this up

        # Reload from disk: the in-memory `mobilenet_frozen` / `efficientnetb0_frozen`
        # objects were mutated in place by their fine-tune siblings, so they no
        # longer carry the original frozen weights. The `.keras` was saved
        # before mutation and is the authoritative copy.
        logger.info(f"Reloading winner from {winner_artefact} and evaluating on test set…")
        winner_model = tf.keras.models.load_model(winner_artefact, compile=False)
        winner_model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
        )
        test_ds = make_dataset(raw_test, winner["preprocessing"], training=False)
        _, test_acc = winner_model.evaluate(test_ds, verbose=0)

        # Per-class P/R/F1 on the test set — winner only, once.
        y_true: list[int] = []
        y_pred: list[int] = []
        for x, y in test_ds:
            y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
            y_pred.extend(np.argmax(winner_model.predict(x, verbose=0), axis=1).tolist())
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        tp_macro, tr_macro, tf1_macro, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        t_prec, t_rec, t_f1, t_support = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
            precision_recall_fscore_support(
                y_true_arr, y_pred_arr, average=None, zero_division=0,
                labels=list(range(NUM_CLASSES)),
            ),
        )

        results["winner_test_accuracy"] = float(test_acc)
        results["winner_test_f1_macro"] = float(tf1_macro)
        results["winner_test_precision_macro"] = float(tp_macro)
        results["winner_test_recall_macro"] = float(tr_macro)
        results["winner_test_per_class"] = [
            {
                "class": class_names[i],
                "precision": float(t_prec[i]),
                "recall": float(t_rec[i]),
                "f1": float(t_f1[i]),
                "support": int(t_support[i]),
            }
            for i in range(NUM_CLASSES)
        ]
        logger.info(
            f"  winner test_accuracy={test_acc:.4f}  test_f1_macro={tf1_macro:.4f}"
        )

        (REPO_ROOT / "model_info.json").write_text(
            json.dumps(
                {
                    "name": winner["name"],
                    "test_accuracy": float(test_acc),
                    "test_f1_macro": float(tf1_macro),
                    "val_accuracy": winner["val_accuracy"],
                    "params_total": winner["params_total"],
                    "model_size_mb": winner["model_size_mb"],
                    "avg_inference_time_ms": winner["avg_inference_time_ms"],
                    "preprocessing": winner["preprocessing"],
                    "gradcam_target_layer": winner["gradcam_target_layer"],
                    "trained_on": "Oxford-IIIT Pet",
                    "input_size": list(IMAGE_SIZE),
                    "num_classes": len(class_names),
                    "run_date": today,
                },
                indent=2,
            )
        )
    else:
        logger.error("No model completed successfully — winner not selected.")

    results_path.write_text(json.dumps(results, indent=2))
    plot_speed_accuracy(results, OUT_ROOT / "plots" / f"speed_vs_accuracy_{today}.png")
    logger.info(f"Done. Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
