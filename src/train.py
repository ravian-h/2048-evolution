"""Evolve heuristic weights for the 2048 agent.

    python src/train.py --generations 30 --population 30 --out best.npy
"""
import argparse

import numpy as np

from ai import FEATURE_NAMES, EvolutionConfig, HeuristicAgent, evolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--games", type=int, default=5, help="games per fitness evaluation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", default="best_weights.npy")
    args = parser.parse_args()

    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        games_per_eval=args.games,
        seed=args.seed,
    )

    def report(stats):
        print(f"gen {stats.generation:3d}  best {stats.best_fitness:8.0f}  mean {stats.mean_fitness:8.0f}")

    result = evolve(config, on_generation=report)
    np.save(args.out, result.best_weights)

    print("\nBest weights:")
    for name, w in zip(FEATURE_NAMES, result.best_weights):
        print(f"  {name:>14}: {w:+.3f}")

    agent = HeuristicAgent(result.best_weights)
    games = [agent.play(seed=10_000 + i) for i in range(20)]
    print(f"\nValidation over 20 fresh games: mean score {np.mean([g.score for g in games]):.0f}, "
          f"max tiles {sorted(g.max_tile() for g in games)}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
