"""A player that picks moves by scoring the resulting boards.

The agent is fully described by its weight vector, which is what the
evolutionary algorithm optimises.
"""
from __future__ import annotations

import numpy as np

from game import Direction, Game2048, apply_move
from .heuristics import NUM_FEATURES, evaluate


class HeuristicAgent:
    def __init__(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (NUM_FEATURES,):
            raise ValueError(f"expected {NUM_FEATURES} weights, got shape {weights.shape}")
        self.weights = weights

    def choose_move(self, board: np.ndarray) -> Direction | None:
        """Greedy one-step look-ahead: maximise heuristic of the board after
        the move (before the random tile spawns)."""
        best_move, best_value = None, -np.inf
        for direction in Direction:
            new_board, _, moved = apply_move(board, direction)
            if not moved:
                continue
            value = evaluate(new_board, self.weights)
            if value > best_value:
                best_move, best_value = direction, value
        return best_move

    def play(self, seed: int | None = None, max_moves: int = 100_000) -> Game2048:
        game = Game2048(seed=seed)
        while game.moves < max_moves:
            move = self.choose_move(game.board)
            if move is None:
                break
            game.move(move)
        return game
