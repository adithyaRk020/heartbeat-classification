from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


@dataclass
class HeartbeatData:
    X_train_flat: np.ndarray
    X_val_flat: np.ndarray
    X_test_flat: np.ndarray
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_onehot: np.ndarray
    y_val_onehot: np.ndarray
    y_test_onehot: np.ndarray
    num_classes: int
    class_names: Tuple[str, ...]


def _is_label_candidate(series: pd.Series, max_classes: int = 20) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    unique_values = np.sort(numeric.unique())
    if unique_values.size < 2 or unique_values.size > max_classes:
        return False
    return np.allclose(unique_values, unique_values.astype(int))


def _infer_label_column(df: pd.DataFrame) -> int:
    candidates = [0, df.shape[1] - 1]
    valid = [idx for idx in candidates if _is_label_candidate(df.iloc[:, idx])]
    if not valid:
        raise ValueError(
            "Unable to infer label column. Expected label at first or last column "
            "with discrete integer-like classes."
        )
    if len(valid) == 1:
        return valid[0]
    uniq_counts = {idx: df.iloc[:, idx].nunique() for idx in valid}
    return min(valid, key=lambda idx: uniq_counts[idx])


def _split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    label_col = _infer_label_column(df)
    y = df.iloc[:, label_col].to_numpy(dtype=np.int32)
    X = df.drop(df.columns[label_col], axis=1).to_numpy(dtype=np.float32)
    return X, y


def load_heartbeat_data(
    data_dir: str | Path,
    train_file: str = "mitbih_train.csv",
    test_file: str = "mitbih_test.csv",
    validation_split: float = 0.2,
    random_state: int = 42,
) -> HeartbeatData:
    data_dir = Path(data_dir)
    train_path = data_dir / train_file
    test_path = data_dir / test_file

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected files at {train_path} and {test_path}. "
            "Download from Kaggle heartbeat dataset and place them in the data directory."
        )

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train_full, y_train_full = _split_features_labels(train_df)
    X_test, y_test = _split_features_labels(test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    X_train_seq = np.expand_dims(X_train, axis=-1)
    X_val_seq = np.expand_dims(X_val, axis=-1)
    X_test_seq = np.expand_dims(X_test, axis=-1)

    class_names = tuple(f"Class {idx}" for idx in range(num_classes))

    return HeartbeatData(
        X_train_flat=X_train,
        X_val_flat=X_val,
        X_test_flat=X_test,
        X_train_seq=X_train_seq,
        X_val_seq=X_val_seq,
        X_test_seq=X_test_seq,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_train_onehot=y_train_onehot,
        y_val_onehot=y_val_onehot,
        y_test_onehot=y_test_onehot,
        num_classes=num_classes,
        class_names=class_names,
    )
