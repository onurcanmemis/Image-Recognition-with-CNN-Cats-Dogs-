"""Model 6 — EfficientNetB0 fine-tuned.

Same recipe as `mobilenet_finetune.py` but for EfficientNetB0: continue from
the trained Model 5, unfreeze the top 20 % of base layers, keep BatchNorm
frozen, recompile at lr=1e-5.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as keras_layers

NAME = "efficientnetb0_finetune"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "top_activation"
PREPROCESSING = "efficientnet"
UNFREEZE_FRACTION = 0.20
FINETUNE_LR = 1e-5


def _efficientnet_base(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if layer.name.startswith("efficientnet"):
            return layer
    raise ValueError("Could not locate EfficientNetB0 base inside the trained frozen model")


def build(trained_frozen_model: tf.keras.Model) -> tf.keras.Model:
    """Unfreeze the top ~20 % of the EfficientNetB0 base and recompile at a low LR."""
    base = _efficientnet_base(trained_frozen_model)
    base.trainable = True

    total = len(base.layers)
    n_unfreeze = int(round(total * UNFREEZE_FRACTION))
    cutoff = total - n_unfreeze
    for i, layer in enumerate(base.layers):
        if i < cutoff:
            layer.trainable = False
        elif isinstance(layer, keras_layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trained_frozen_model._name = NAME
    trained_frozen_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return trained_frozen_model
