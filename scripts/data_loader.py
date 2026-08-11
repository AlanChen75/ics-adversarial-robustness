"""
Data loading and preprocessing for SWaT, WADI, and SMD datasets.
Handles downloading, normalization, windowing, and train/test splitting.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# WADI attack periods (row ranges in WADI_attackdataLABLE.csv)
# Source: iTrust WADI documentation — 15 attack scenarios
# ---------------------------------------------------------------------------
WADI_ATTACK_RANGES: list[tuple[int, int]] = [
    (7866, 7937),      # Attack 1
    (8466, 8538),      # Attack 2
    (38714, 38785),     # Attack 3
    (45714, 45785),     # Attack 4
    (62914, 62985),     # Attack 5
    (66114, 66185),     # Attack 6
    (67514, 67585),     # Attack 7
    (73514, 73585),     # Attack 8
    (83014, 83085),     # Attack 9
    (98814, 98885),     # Attack 10
    (102014, 102085),   # Attack 11
    (103614, 103685),   # Attack 12
    (113214, 113285),   # Attack 13
    (131014, 131085),   # Attack 14
    (150714, 150785),   # Attack 15
]


class ICSDataset(Dataset):
    """PyTorch Dataset for ICS time series windows."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Helper: robust CSV loading with delimiter detection
# ---------------------------------------------------------------------------

def _read_csv_robust(path: str, skiprows: int = 0) -> pd.DataFrame:
    """Read a CSV file, auto-detecting semicolon vs comma delimiter.

    Tries comma first; falls back to semicolon if few columns are detected.
    Also handles multi-row headers (skiprows) common in WADI files.
    """
    df = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    if len(df.columns) <= 2:
        df = pd.read_csv(path, skiprows=skiprows, sep=";", low_memory=False)
    return df


def _detect_wadi_skiprows(path: str) -> int:
    """Detect how many header rows to skip in a WADI CSV file.

    WADI files from the MAD-GAN Google Drive often have 0-4 metadata lines
    before the actual CSV header.  We peek at the first 10 lines and pick
    the row that looks like a proper header (many comma-separated fields).
    """
    with open(path, "r", errors="replace") as f:
        lines = [f.readline() for _ in range(10)]

    for idx, line in enumerate(lines):
        # A real CSV header has many commas
        if line.count(",") >= 5:
            return idx
    return 0


def _shorten_wadi_columns(columns: pd.Index) -> list[str]:
    r"""Strip the long Windows path prefix from WADI column names.

    Raw names look like:  \\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\...
    We keep only the final segment after the last backslash.
    """
    short: list[str] = []
    for col in columns:
        col_str = str(col).strip()
        if "\\" in col_str:
            col_str = col_str.rsplit("\\", maxsplit=1)[-1]
        short.append(col_str)
    return short


def _generate_wadi_labels(n_rows: int) -> np.ndarray:
    """Generate binary labels for WADI attack data from known attack ranges.

    Returns a 1-D int array of length *n_rows* where 1 = attack, 0 = normal.
    """
    labels = np.zeros(n_rows, dtype=int)
    for start, end in WADI_ATTACK_RANGES:
        # Clamp to actual data length (ranges are 0-indexed row numbers)
        lo = min(start, n_rows)
        hi = min(end + 1, n_rows)  # inclusive end
        labels[lo:hi] = 1
    return labels


# ---------------------------------------------------------------------------
# SWaT loader
# ---------------------------------------------------------------------------

def load_swat(
    data_dir: str,
    window_size: int = 100,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and preprocess SWaT dataset.

    Expected files in *data_dir*:
    - SWaT_Dataset_Normal_v1.csv  (training)
    - SWaT_Dataset_Attack_v0.csv  (testing)
    """
    train_path = os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")
    test_path = os.path.join(data_dir, "SWaT_Dataset_Attack_v0.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, "swat_train.csv")
        test_path = os.path.join(data_dir, "swat_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"SWaT data not found in {data_dir}. "
            "Download from https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/"
        )

    print(f"Loading SWaT training data from {train_path}")
    train_df = _read_csv_robust(train_path)

    print(f"Loading SWaT test data from {test_path}")
    test_df = _read_csv_robust(test_path)

    # --- Extract labels --------------------------------------------------
    if "Normal/Attack" in test_df.columns:
        test_labels = (
            test_df["Normal/Attack"].str.strip() == "Attack"
        ).astype(int).values
        train_df = train_df.drop(
            columns=["Normal/Attack", "Timestamp"], errors="ignore"
        )
        test_df = test_df.drop(
            columns=["Normal/Attack", "Timestamp"], errors="ignore"
        )
    elif "label" in test_df.columns:
        test_labels = test_df["label"].values
        train_df = train_df.drop(
            columns=["label", "timestamp"], errors="ignore"
        )
        test_df = test_df.drop(
            columns=["label", "timestamp"], errors="ignore"
        )
    else:
        # Assume last column is label
        test_labels = test_df.iloc[:, -1].values
        train_df = train_df.iloc[:, 1:-1]
        test_df = test_df.iloc[:, 1:-1]

    # --- Numeric only + align columns -----------------------------------
    train_df = train_df.select_dtypes(include=[np.number])
    test_df = test_df.select_dtypes(include=[np.number])

    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    # --- Remove constant columns ----------------------------------------
    std = train_df.std()
    non_constant = std[std > 1e-6].index.tolist()
    train_df = train_df[non_constant]
    test_df = test_df[non_constant]

    # --- Fill NaN -------------------------------------------------------
    train_df = train_df.ffill().fillna(0)
    test_df = test_df.ffill().fillna(0)

    n_features = train_df.shape[1]
    print(
        f"SWaT: {n_features} features, "
        f"{len(train_df)} train, {len(test_df)} test samples"
    )

    # --- Normalize & window ---------------------------------------------
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df.values)
    test_data = scaler.transform(test_df.values)

    train_windows = create_windows(train_data, window_size, stride)
    test_windows, test_window_labels = create_windows_with_labels(
        test_data, test_labels, window_size, stride
    )

    metadata: dict[str, Any] = {
        "dataset": "swat",
        "n_features": n_features,
        "feature_names": non_constant,
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "attack_ratio": float(test_window_labels.mean()),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }

    return train_windows, test_windows, test_window_labels, metadata


# ---------------------------------------------------------------------------
# WADI loader
# ---------------------------------------------------------------------------

def load_wadi(
    data_dir: str,
    window_size: int = 100,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and preprocess WADI dataset.

    Expected files in *data_dir*:
    - WADI_14days_new.csv          (training — ~1.2 M rows)
    - WADI_attackdataLABLE.csv     (testing  — ~170 K rows)

    The attack file from the MAD-GAN Google Drive typically does **not**
    contain a label column.  Labels are generated from the 15 known attack
    time-ranges published in the iTrust documentation.

    Parameters
    ----------
    stride : int
        Default 5 (instead of 1) to keep memory usage manageable on WADI
        which has ~1.2 M training rows.
    """
    train_path = os.path.join(data_dir, "WADI_14days_new.csv")
    test_path = os.path.join(data_dir, "WADI_attackdataLABLE.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, "wadi_train.csv")
        test_path = os.path.join(data_dir, "wadi_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"WADI data not found in {data_dir}. "
            "Download from https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_wadi/"
        )

    # --- Load training data ---------------------------------------------
    print(f"Loading WADI training data from {train_path}")
    skip_train = _detect_wadi_skiprows(train_path)
    train_df = pd.read_csv(train_path, skiprows=skip_train, low_memory=False)
    train_df.columns = _shorten_wadi_columns(train_df.columns)

    # --- Load test data -------------------------------------------------
    print(f"Loading WADI test data from {test_path}")
    skip_test = _detect_wadi_skiprows(test_path)
    test_df = pd.read_csv(test_path, skiprows=skip_test, low_memory=False)
    test_df.columns = _shorten_wadi_columns(test_df.columns)

    # --- Extract or generate labels -------------------------------------
    label_cols = [
        c for c in test_df.columns
        if "label" in c.lower() or "attack" in c.lower()
    ]

    if label_cols:
        print(f"  Found label column(s): {label_cols}")
        test_labels = test_df[label_cols[0]].fillna(0)
        # Coerce to numeric (some files have string labels)
        test_labels = pd.to_numeric(test_labels, errors="coerce").fillna(0)
        # Make binary: any non-zero value → 1 (attack)
        test_labels = (test_labels != 0).astype(int).values
    else:
        print(
            "  No label column found — generating labels from "
            f"{len(WADI_ATTACK_RANGES)} known attack time-ranges"
        )
        test_labels = _generate_wadi_labels(len(test_df))
        n_attack = int(test_labels.sum())
        print(
            f"  Generated labels: {n_attack} attack rows "
            f"({n_attack / len(test_df) * 100:.2f}%) out of {len(test_df)}"
        )

    # --- Drop non-feature columns ---------------------------------------
    drop_patterns = {"row", "date", "time"}
    drop_cols = label_cols + [
        c for c in test_df.columns
        if c.lower().strip() in drop_patterns
    ]
    train_df = train_df.drop(
        columns=[c for c in drop_cols if c in train_df.columns],
        errors="ignore",
    )
    test_df = test_df.drop(
        columns=[c for c in drop_cols if c in test_df.columns],
        errors="ignore",
    )

    # --- Numeric only ---------------------------------------------------
    train_df = train_df.select_dtypes(include=[np.number])
    test_df = test_df.select_dtypes(include=[np.number])

    # --- Align columns --------------------------------------------------
    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not common_cols:
        raise ValueError(
            "No common numeric columns between WADI train and test files. "
            "Check that both CSVs use the same column naming scheme."
        )
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    # --- Remove constant columns ----------------------------------------
    std = train_df.std()
    non_constant = std[std > 1e-6].index.tolist()
    train_df = train_df[non_constant]
    test_df = test_df[non_constant]

    # --- Fill NaN (forward fill, then zero) -----------------------------
    train_df = train_df.ffill().fillna(0)
    test_df = test_df.ffill().fillna(0)

    n_features = train_df.shape[1]
    print(
        f"WADI: {n_features} features, "
        f"{len(train_df)} train, {len(test_df)} test samples "
        f"(stride={stride})"
    )

    # --- Normalize & window ---------------------------------------------
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df.values)
    test_data = scaler.transform(test_df.values)

    train_windows = create_windows(train_data, window_size, stride)
    test_windows, test_window_labels = create_windows_with_labels(
        test_data, test_labels, window_size, stride
    )

    metadata: dict[str, Any] = {
        "dataset": "wadi",
        "n_features": n_features,
        "feature_names": non_constant,
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "attack_ratio": float(test_window_labels.mean()),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }

    return train_windows, test_windows, test_window_labels, metadata


# ---------------------------------------------------------------------------
# Windowing utilities
# ---------------------------------------------------------------------------

def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Create sliding windows from time series data."""
    n_samples = max(0, (len(data) - window_size) // stride + 1)
    windows = np.zeros((n_samples, window_size, data.shape[1]))
    for i in range(n_samples):
        start = i * stride
        windows[i] = data[start : start + window_size]
    return windows


def create_windows_with_labels(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows with labels.

    A window is labelled *attack* (1) if **any** point in the window is attack.
    """
    n_samples = max(0, (len(data) - window_size) // stride + 1)
    windows = np.zeros((n_samples, window_size, data.shape[1]))
    window_labels = np.zeros(n_samples)
    for i in range(n_samples):
        start = i * stride
        windows[i] = data[start : start + window_size]
        window_labels[i] = (
            1.0 if labels[start : start + window_size].any() else 0.0
        )
    return windows, window_labels


# ---------------------------------------------------------------------------
# SMD loader
# ---------------------------------------------------------------------------

def load_smd(
    data_dir: str,
    window_size: int = 100,
    stride: int = 1,
    machine: str = "machine-1-1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and preprocess Server Machine Dataset (SMD).

    SMD has 28 machines, each with 38 features.
    Format: CSV-like .txt files with comma-separated values.
    """
    train_path = os.path.join(data_dir, "train", f"{machine}.txt")
    test_path = os.path.join(data_dir, "test", f"{machine}.txt")
    label_path = os.path.join(data_dir, "test_label", f"{machine}.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"SMD data not found at {train_path}. "
            "Clone from https://github.com/NetManAIOps/OmniAnomaly"
        )

    print(f"Loading SMD {machine} from {data_dir}")
    train_data = np.loadtxt(train_path, delimiter=",")
    test_data = np.loadtxt(test_path, delimiter=",")
    test_labels = np.loadtxt(label_path, delimiter=",").astype(int)

    # Handle multi-column labels
    if test_labels.ndim > 1:
        test_labels = test_labels.any(axis=1).astype(int)

    n_features = train_data.shape[1]
    print(
        f"SMD {machine}: {n_features} features, "
        f"{len(train_data)} train, {len(test_data)} test, "
        f"attack ratio: {test_labels.mean():.3f}"
    )

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_windows = create_windows(train_data, window_size, stride)
    test_windows, test_window_labels = create_windows_with_labels(
        test_data, test_labels, window_size, stride
    )

    metadata: dict[str, Any] = {
        "dataset": f"smd_{machine}",
        "n_features": n_features,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "attack_ratio": float(test_window_labels.mean()),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }

    return train_windows, test_windows, test_window_labels, metadata


# ---------------------------------------------------------------------------
# Unified dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    window_size: int = 100,
    stride: int = 1,
    batch_size: int = 256,
    val_ratio: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Get train, val, test DataLoaders for a dataset.

    For WADI the default stride is overridden to 5 (unless the caller
    explicitly passes a different value) to keep memory usage reasonable.
    """
    # Use a larger default stride for WADI if caller left it at 1
    effective_stride = stride
    if dataset_name == "wadi" and stride == 1:
        effective_stride = 5
        print(
            "WADI: auto-increasing stride from 1 → 5 to reduce memory. "
            "Pass stride explicitly to override."
        )

    if dataset_name == "swat":
        train_windows, test_windows, test_labels, metadata = load_swat(
            data_dir, window_size, effective_stride
        )
    elif dataset_name == "wadi":
        train_windows, test_windows, test_labels, metadata = load_wadi(
            data_dir, window_size, effective_stride
        )
    elif dataset_name == "smd":
        train_windows, test_windows, test_labels, metadata = load_smd(
            data_dir, window_size, effective_stride
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Train / val split ----------------------------------------------
    n_val = int(len(train_windows) * val_ratio)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_windows))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_set = ICSDataset(
        train_windows[train_indices],
        np.zeros(len(train_indices)),  # all normal
    )
    val_set = ICSDataset(
        train_windows[val_indices],
        np.zeros(len(val_indices)),
    )
    test_set = ICSDataset(test_windows, test_labels)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, metadata
