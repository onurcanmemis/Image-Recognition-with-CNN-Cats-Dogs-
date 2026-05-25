"""Model 4 — EfficientNetB0 with the convolutional base frozen."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

NAME = "efficientnetb0_frozen"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "top_activation"    # last activation in EfficientNetB0
PREPROCESSING = "efficientnet"             # tf.keras.applications.efficientnet.preprocess_input


def build(num_classes: int = 37) -> tf.keras.Model:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    # Call the base as a sub-model so it stays a single named layer in the outer
    # Model — the fine-tune builder relies on get_layer("efficientnetb0").
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs, output, name=NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
