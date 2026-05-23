"""Model 2 — Deeper CNN.

Three Conv/MaxPool/BatchNorm blocks → Dropout → Dense(256) → softmax(37).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential, layers

NAME = "deeper_cnn"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "last_conv"
PREPROCESSING = "rescale_1_over_255"


def build(num_classes: int = 37) -> tf.keras.Model:
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", name="last_conv"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ], name=NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
