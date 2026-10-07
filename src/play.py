"""Play 2048 in the terminal, or watch an agent play.

    python src/play.py                       # play yourself with w/a/s/d
    python src/play.py --weights best.npy    # watch an evolved agent
"""
import argparse

import numpy as np

from ai import HeuristicAgent
from game import Direction, Game2048

KEYS = {"w": Direction.UP, "a": Direction.LEFT, "s": Direction.DOWN, "d": Direction.RIGHT}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="path to a .npy weight vector saved by train.py")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.weights:
        game = HeuristicAgent(np.load(args.weights)).play(seed=args.seed)
        print(game)
        print(f"Moves: {game.moves}  Max tile: {game.max_tile()}")
        return

    game = Game2048(seed=args.seed)
    while not game.is_over():
        print(game)
        key = input("move (w/a/s/d, q to quit): ").strip().lower()
        if key == "q":
            return
        if key not in KEYS or not game.move(KEYS[key]):
            print("Invalid move.")
    print(game)
    print(f"Game over! Max tile: {game.max_tile()}")


if __name__ == "__main__":
    main()
