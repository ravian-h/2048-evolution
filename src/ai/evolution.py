"""Genetic algorithm that evolves the agent's heuristic weights.

Each individual is a weight vector. Fitness is the mean score over a few
games; every individual in a generation plays the same seeds so they are
compared on equal terms, and the seeds change every generation so the
population can't overfit to particular games.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent import HeuristicAgent
from .heuristics import NUM_FEATURES


@dataclass
class EvolutionConfig:
    population_size: int = 30
    generations: int = 30
    games_per_eval: int = 5
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.3      # chance each gene is mutated
    mutation_scale: float = 0.5     # std-dev of gaussian mutation
    init_scale: float = 2.0
    seed: int | None = None


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_weights: np.ndarray


@dataclass
class EvolutionResult:
    best_weights: np.ndarray
    best_fitness: float
    history: list[GenerationStats] = field(default_factory=list)


def fitness(weights: np.ndarray, seeds) -> float:
    agent = HeuristicAgent(weights)
    return float(np.mean([agent.play(seed=int(s)).score for s in seeds]))


def _tournament(rng, population, fitnesses, size):
    contestants = rng.choice(len(population), size=size, replace=False)
    return population[contestants[np.argmax(fitnesses[contestants])]]


def _crossover(rng, a, b):
    """Blend crossover: each gene is a random interpolation of the parents."""
    alpha = rng.random(a.shape)
    return alpha * a + (1 - alpha) * b


def _mutate(rng, weights, rate, scale):
    mask = rng.random(weights.shape) < rate
    return weights + mask * rng.normal(0, scale, weights.shape)


def evolve(config: EvolutionConfig = EvolutionConfig(), on_generation=None) -> EvolutionResult:
    rng = np.random.default_rng(config.seed)
    population = rng.normal(0, config.init_scale, (config.population_size, NUM_FEATURES))
    result = EvolutionResult(best_weights=population[0].copy(), best_fitness=-np.inf)

    for gen in range(config.generations):
        seeds = rng.integers(0, 2**31, config.games_per_eval)
        fitnesses = np.array([fitness(ind, seeds) for ind in population])

        order = np.argsort(fitnesses)[::-1]
        stats = GenerationStats(gen, float(fitnesses[order[0]]), float(fitnesses.mean()), population[order[0]].copy())
        result.history.append(stats)
        # Note: best_fitness is measured on that generation's seeds, so it's
        # a noisy estimate; re-evaluate the final weights on fresh games.
        if stats.best_fitness > result.best_fitness:
            result.best_fitness = stats.best_fitness
            result.best_weights = stats.best_weights
        if on_generation:
            on_generation(stats)

        next_population = [population[i].copy() for i in order[: config.elite_count]]
        while len(next_population) < config.population_size:
            a = _tournament(rng, population, fitnesses, config.tournament_size)
            b = _tournament(rng, population, fitnesses, config.tournament_size)
            child = _mutate(rng, _crossover(rng, a, b), config.mutation_rate, config.mutation_scale)
            next_population.append(child)
        population = np.array(next_population)

    return result
