from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from data_loading import HeartbeatData, load_heartbeat_data
from evaluate import evaluate_model
from models import build_residual_mlp, build_tcn_cnn_attention
from visualize import plot_training_curves


def _compile_and_train(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def _save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    history_payload = {key: [float(x) for x in values] for key, values in history.history.items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)


def _train_residual_mlp(data: HeartbeatData, output_dir: Path, args) -> None:
    model_dir = output_dir / "residual_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_residual_mlp(
        input_dim=data.X_train_flat.shape[1],
        num_classes=data.num_classes,
    )

    history = _compile_and_train(
        model,
        data.X_train_flat,
        data.y_train_onehot,
        data.X_val_flat,
        data.y_val_onehot,
        model_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    _save_history(history, model_dir / "history.json")
    plot_training_curves(history.history, model_dir, "Residual MLP")
    evaluate_model(
        model,
        data.X_test_flat,
        data.y_test,
        data.class_names,
        model_dir,
        model_name="Residual MLP",
    )


def _train_tcn_cnn_attention(data: HeartbeatData, output_dir: Path, args) -> None:
    model_dir = output_dir / "tcn_cnn_attention"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_tcn_cnn_attention(
        input_shape=data.X_train_seq.shape[1:],
        num_classes=data.num_classes,
    )

    history = _compile_and_train(
        model,
        data.X_train_seq,
        data.y_train_onehot,
        data.X_val_seq,
        data.y_val_onehot,
        model_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    _save_history(history, model_dir / "history.json")
    plot_training_curves(history.history, model_dir, "TCN CNN Attention")
    evaluate_model(
        model,
        data.X_test_seq,
        data.y_test,
        data.class_names,
        model_dir,
        model_name="TCN CNN Attention",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heartbeat classification models")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing mitbih_train.csv and mitbih_test.csv")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where model artifacts and plots are written")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.random_state)

    data = load_heartbeat_data(
        data_dir=args.data_dir,
        validation_split=args.validation_split,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_residual_mlp(data, output_dir, args)
    _train_tcn_cnn_attention(data, output_dir, args)


if __name__ == "__main__":
    main()
