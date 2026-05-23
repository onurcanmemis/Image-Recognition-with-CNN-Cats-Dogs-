"""Model 1 — Baseline CNN.

Two Conv/MaxPool blocks → Flatten → Dense(128) → softmax(37).
Trained from scratch on Oxford-IIIT Pet at 224×224.

Output layer is explicit `float32` so mixed-precision (fp16 elsewhere) doesn't
destabilise the softmax / cross-entropy.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential, layers

NAME = "baseline_cnn"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "last_conv"   # see Conv2D name='last_conv' below
PREPROCESSING = "rescale_1_over_255"


def build(num_classes: int = 37) -> tf.keras.Model:
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", name="last_conv"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ], name=NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
