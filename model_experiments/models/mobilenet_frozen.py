"""Model 3 — MobileNetV2 with the convolutional base frozen.

Only the new classification head trains. Cheap, fast, strong baseline for
transfer learning on small datasets.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2

NAME = "mobilenet_frozen"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "out_relu"          # last activation in MobileNetV2
PREPROCESSING = "mobilenet_v2"             # tf.keras.applications.mobilenet_v2.preprocess_input


def build(num_classes: int = 37) -> tf.keras.Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(base.input, output, name=NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
