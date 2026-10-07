"""Board features the agent combines into a single score.

All features work on log2 tile values so a 2048 tile counts 11, not 2048;
otherwise the largest tile would dominate everything.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("empty", "monotonicity", "smoothness", "max_in_corner", "merges")
NUM_FEATURES = len(FEATURE_NAMES)


def _log2(board: np.ndarray) -> np.ndarray:
    logs = np.zeros(board.shape, dtype=np.float64)
    nonzero = board > 0
    logs[nonzero] = np.log2(board[nonzero])
    return logs


def features(board: np.ndarray) -> np.ndarray:
    logs = _log2(board)

    empty = float((board == 0).sum())

    # Monotonicity: penalise rows/columns that go up and down instead of
    # steadily increasing in one direction. 0 is perfect, more negative is worse.
    row_diffs = np.diff(logs, axis=1)
    col_diffs = np.diff(logs, axis=0)
    mono_rows = -np.minimum(np.clip(row_diffs, 0, None).sum(axis=1), -np.clip(row_diffs, None, 0).sum(axis=1)).sum()
    mono_cols = -np.minimum(np.clip(col_diffs, 0, None).sum(axis=0), -np.clip(col_diffs, None, 0).sum(axis=0)).sum()
    monotonicity = mono_rows + mono_cols

    # Smoothness: neighbouring tiles of similar size are easy to merge later.
    # Only compare non-empty pairs, so empty cells don't count as "rough".
    occupied = board > 0
    h_pairs = occupied[:, :-1] & occupied[:, 1:]
    v_pairs = occupied[:-1, :] & occupied[1:, :]
    smoothness = -(np.abs(row_diffs)[h_pairs].sum() + np.abs(col_diffs)[v_pairs].sum())

    max_value = logs.max()
    corners = (logs[0, 0], logs[0, -1], logs[-1, 0], logs[-1, -1])
    max_in_corner = max_value if max_value in corners else 0.0

    merges = float(((board[:, :-1] == board[:, 1:]) & h_pairs).sum() + ((board[:-1, :] == board[1:, :]) & v_pairs).sum())

    return np.array([empty, monotonicity, smoothness, max_in_corner, merges])


def evaluate(board: np.ndarray, weights: np.ndarray) -> float:
    return float(features(board) @ weights)
