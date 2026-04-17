from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)


def build_residual_mlp(
    input_dim: int,
    num_classes: int,
    hidden_units: Sequence[int] = (256, 128, 64),
    dropout_rate: float = 0.3,
) -> Model:
    inputs = Input(shape=(input_dim,), name="ecg_input_flat")
    x = inputs

    for units in hidden_units:
        residual = x
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)

        if residual.shape[-1] != units:
            residual = Dense(units)(residual)
        x = Add()([x, residual])
        x = Activation("relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="classification_head")(x)
    return Model(inputs=inputs, outputs=outputs, name="residual_mlp")


def _tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
) -> tf.Tensor:
    residual = x

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = Conv1D(filters, kernel_size=1, padding="same")(residual)

    x = Add()([x, residual])
    return Activation("relu")(x)


def build_tcn_cnn_attention(
    input_shape: Sequence[int],
    num_classes: int,
    tcn_filters: int = 64,
    tcn_kernel_size: int = 3,
    dilations: Iterable[int] = (1, 2, 4, 8),
    dropout_rate: float = 0.2,
) -> Model:
    inputs = Input(shape=input_shape, name="ecg_input_seq")

    x = inputs
    for dilation in dilations:
        x = _tcn_residual_block(
            x,
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
        )

    cnn_3 = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    cnn_5 = Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    fused = Concatenate()([cnn_3, cnn_5])

    attended = MultiHeadAttention(num_heads=4, key_dim=32, dropout=dropout_rate)(
        fused, fused
    )
    fused = Add()([fused, attended])
    fused = LayerNormalization()(fused)

    pooled = GlobalAveragePooling1D()(fused)
    dense = Dense(128, activation="relu")(pooled)
    dense = Dropout(dropout_rate)(dense)
    outputs = Dense(num_classes, activation="softmax", name="classification_head")(dense)

    return Model(inputs=inputs, outputs=outputs, name="tcn_cnn_attention")
