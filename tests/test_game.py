import numpy as np
import pytest

from game import Direction, Game2048, apply_move, has_moves, legal_moves, merge_row_left


@pytest.mark.parametrize("row, expected, gained", [
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([0, 2, 0, 2], [4, 0, 0, 0], 4),
    ([2, 2, 2, 0], [4, 2, 0, 0], 4),
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([4, 4, 8, 0], [8, 8, 0, 0], 8),   # a merged tile doesn't merge again
    ([2, 4, 8, 16], [2, 4, 8, 16], 0),
    ([8, 0, 8, 4], [16, 4, 0, 0], 16),
])
def test_merge_row_left(row, expected, gained):
    assert merge_row_left(row) == (expected, gained)


BOARD = np.array([
    [2, 0, 2, 4],
    [0, 0, 0, 0],
    [2, 0, 0, 4],
    [0, 8, 0, 0],
])


@pytest.mark.parametrize("direction, expected, gained", [
    (Direction.LEFT, [[4, 4, 0, 0], [0, 0, 0, 0], [2, 4, 0, 0], [8, 0, 0, 0]], 4),
    (Direction.RIGHT, [[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 2, 4], [0, 0, 0, 8]], 4),
    (Direction.UP, [[4, 8, 2, 8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 12),
    (Direction.DOWN, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 8, 2, 8]], 12),
])
def test_apply_move(direction, expected, gained):
    new_board, points, moved = apply_move(BOARD, direction)
    assert new_board.tolist() == expected
    assert points == gained
    assert moved


def test_apply_move_does_not_mutate_input():
    board = BOARD.copy()
    apply_move(board, Direction.UP)
    assert np.array_equal(board, BOARD)


def test_illegal_move_detected():
    board = np.array([[2, 4, 0, 0]] + [[0] * 4] * 3)
    assert not apply_move(board, Direction.LEFT)[2]
    assert Direction.LEFT not in legal_moves(board)


def test_full_board_without_merges_is_over():
    board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not has_moves(board)
    assert Game2048(board=board).is_over()


def test_full_board_with_merge_is_not_over():
    board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 4]])
    assert has_moves(board)


def test_new_game_has_two_tiles():
    game = Game2048(seed=1)
    assert (game.board > 0).sum() == 2
    assert set(game.board[game.board > 0]) <= {2, 4}


def test_move_spawns_tile_and_scores():
    game = Game2048(seed=0, board=[[2, 2, 0, 0]] + [[0] * 4] * 3)
    assert game.move(Direction.LEFT)
    assert game.score == 4
    assert game.board[0, 0] == 4
    assert (game.board > 0).sum() == 2


def test_illegal_move_changes_nothing():
    board = [[2, 4, 0, 0]] + [[0] * 4] * 3
    game = Game2048(seed=0, board=board)
    assert not game.move(Direction.LEFT)
    assert game.board.tolist() == board
    assert game.moves == 0


def test_seed_is_reproducible():
    a, b = Game2048(seed=42), Game2048(seed=42)
    for d in [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN] * 5:
        a.move(d)
        b.move(d)
    assert np.array_equal(a.board, b.board) and a.score == b.score
