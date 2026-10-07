"""Core 2048 mechanics.

The board is a 4x4 numpy array of tile values (0 = empty). Every move is
implemented as a "slide left" on a re-oriented view of the board, so there
is only one place where the merge rules live.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

SIZE = 4
SPAWN_FOUR_PROBABILITY = 0.1


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def merge_row_left(row) -> tuple[list[int], int]:
    """Slide one row to the left and merge equal neighbours.

    Each tile merges at most once per move, and merges resolve from the
    side being moved towards: [2, 2, 2, 0] -> [4, 2, 0, 0].
    Returns (new_row, points gained).
    """
    tiles = [int(v) for v in row if v]
    merged: list[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            value = tiles[i] * 2
            merged.append(value)
            gained += value
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (len(row) - len(merged)))
    return merged, gained


def _to_left(board: np.ndarray, direction: Direction) -> np.ndarray:
    """Re-orient the board so `direction` becomes a move to the left."""
    if direction == Direction.LEFT:
        return board
    if direction == Direction.RIGHT:
        return np.fliplr(board)
    if direction == Direction.UP:
        return board.T
    return np.fliplr(board.T)  # DOWN


def _from_left(board: np.ndarray, direction: Direction) -> np.ndarray:
    """Inverse of `_to_left`."""
    if direction == Direction.LEFT:
        return board
    if direction == Direction.RIGHT:
        return np.fliplr(board)
    if direction == Direction.UP:
        return board.T
    return np.fliplr(board).T  # DOWN


def apply_move(board: np.ndarray, direction: Direction) -> tuple[np.ndarray, int, bool]:
    """Apply a move without spawning a tile.

    Pure function, used by both the game and the AI's look-ahead.
    Returns (new_board, points gained, whether anything changed).
    """
    oriented = _to_left(board, direction)
    result = np.zeros_like(oriented)
    gained = 0
    for r in range(oriented.shape[0]):
        result[r], row_gain = merge_row_left(oriented[r])
        gained += row_gain
    new_board = np.ascontiguousarray(_from_left(result, direction))
    return new_board, gained, not np.array_equal(new_board, board)


def legal_moves(board: np.ndarray) -> list[Direction]:
    return [d for d in Direction if apply_move(board, d)[2]]


def has_moves(board: np.ndarray) -> bool:
    if not board.all():
        return True
    # Board is full: a move exists only if two equal tiles are adjacent.
    return bool((board[:, :-1] == board[:, 1:]).any() or (board[:-1, :] == board[1:, :]).any())


class Game2048:
    """A single game: board state, score and random tile spawning."""

    def __init__(self, seed: int | None = None, board: np.ndarray | None = None):
        self.rng = np.random.default_rng(seed)
        self.score = 0
        self.moves = 0
        if board is not None:
            self.board = np.array(board, dtype=np.int64)
        else:
            self.board = np.zeros((SIZE, SIZE), dtype=np.int64)
            self.spawn_tile()
            self.spawn_tile()

    def spawn_tile(self) -> bool:
        """Place a 2 (90%) or 4 (10%) on a random empty cell."""
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return False
        r, c = empty[self.rng.integers(len(empty))]
        self.board[r, c] = 4 if self.rng.random() < SPAWN_FOUR_PROBABILITY else 2
        return True

    def move(self, direction: Direction) -> bool:
        """Play a move. Returns False (and changes nothing) if it is illegal."""
        new_board, gained, moved = apply_move(self.board, direction)
        if not moved:
            return False
        self.board = new_board
        self.score += gained
        self.moves += 1
        self.spawn_tile()
        return True

    def legal_moves(self) -> list[Direction]:
        return legal_moves(self.board)

    def is_over(self) -> bool:
        return not has_moves(self.board)

    def max_tile(self) -> int:
        return int(self.board.max())

    def __str__(self) -> str:
        rows = [" ".join(f"{v:>5}" if v else "    ." for v in row) for row in self.board]
        return f"Score: {self.score}\n" + "\n".join(rows)
