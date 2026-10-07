import numpy as np

from ai import EvolutionConfig, HeuristicAgent, evolve, features
from ai.heuristics import NUM_FEATURES


def test_features_shape_and_empty_count():
    board = np.array([[2, 0, 0, 0]] + [[0] * 4] * 3)
    f = features(board)
    assert f.shape == (NUM_FEATURES,)
    assert f[0] == 15


def test_monotonic_board_scores_better_than_zigzag():
    mono = np.array([[16, 8, 4, 2]] * 4)
    zigzag = np.array([[16, 2, 8, 4]] * 4)
    assert features(mono)[1] > features(zigzag)[1]


def test_agent_finishes_game():
    game = HeuristicAgent(np.ones(NUM_FEATURES)).play(seed=0)
    assert game.is_over()
    assert game.score > 0


def test_evolve_runs_and_tracks_history():
    config = EvolutionConfig(population_size=6, generations=2, games_per_eval=1, seed=0)
    result = evolve(config)
    assert len(result.history) == 2
    assert result.best_weights.shape == (NUM_FEATURES,)
    assert result.best_fitness == max(s.best_fitness for s in result.history)
