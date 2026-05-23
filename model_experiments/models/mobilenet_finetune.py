"""Model 4 — MobileNetV2 fine-tuned.

Continues from the trained Model 3 (frozen base + trained head). The top 20 %
of the MobileNetV2 base is unfrozen, BatchNorm layers stay frozen (changing
their running stats during fine-tuning is a classic foot-gun), and the model
is recompiled at lr=1e-5 — CRITICAL: any higher and the pretrained weights
get destroyed.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as keras_layers

NAME = "mobilenet_finetune"
INPUT_SHAPE = (224, 224, 3)
GRADCAM_TARGET_LAYER = "out_relu"
PREPROCESSING = "mobilenet_v2"
UNFREEZE_FRACTION = 0.20
FINETUNE_LR = 1e-5


def _mobilenet_base(model: tf.keras.Model) -> tf.keras.Model:
    """Pull the MobileNetV2 base out of a head-only-trained model."""
    for layer in model.layers:
        if layer.name == "mobilenetv2_1.00_224" or layer.name.startswith("mobilenetv2"):
            return layer
    # Fallback: the base is whatever is *not* one of the head layers.
    raise ValueError("Could not locate MobileNetV2 base inside the trained frozen model")


def build(trained_frozen_model: tf.keras.Model) -> tf.keras.Model:
    """Unfreeze the top ~20 % of the MobileNetV2 base and recompile at a low LR."""
    base = _mobilenet_base(trained_frozen_model)
    base.trainable = True

    total = len(base.layers)
    n_unfreeze = int(round(total * UNFREEZE_FRACTION))
    cutoff = total - n_unfreeze
    for i, layer in enumerate(base.layers):
        if i < cutoff:
            layer.trainable = False
        elif isinstance(layer, keras_layers.BatchNormalization):
            # Freeze BN even in the unfrozen tail; otherwise its running stats drift.
            layer.trainable = False
        else:
            layer.trainable = True

    trained_frozen_model._name = NAME  # rename for clean checkpoint paths
    trained_frozen_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return trained_frozen_model
