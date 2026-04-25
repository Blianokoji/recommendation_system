"""
Genetic Algorithm Weight Optimizer
-----------------------------------
Evolves optimal weights for the 4-component scoring formula:

    Score = W_semantic * Sim + W_cluster * Coherence
          + W_temporal * mu_time + W_actor * mu_actor

Uses pure numpy — no external GA library required.

Chromosome: [W_semantic, W_cluster, W_temporal, W_actor]
Constraint: all weights sum to 1.0, each in [0.01, 0.95]
"""

import numpy as np
from typing import List, Tuple, Callable

# ------------------------------------------------
# GA PARAMETERS
# ------------------------------------------------

POPULATION_SIZE = 50
NUM_GENERATIONS = 30
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
MUTATION_SIGMA = 0.05
NUM_GENES = 4  # [W_semantic, W_cluster, W_temporal, W_actor]

# ------------------------------------------------
# CHROMOSOME UTILITIES
# ------------------------------------------------

def _normalize_chromosome(chromo: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1.0, clamp to [0.01, 0.95]."""
    chromo = np.clip(chromo, 0.01, 0.95)
    return chromo / chromo.sum()


def _init_population(size: int) -> np.ndarray:
    """Initialize a random population of weight vectors."""
    pop = np.random.dirichlet(np.ones(NUM_GENES), size=size)
    return np.array([_normalize_chromosome(c) for c in pop])


# ------------------------------------------------
# SELECTION: Tournament
# ------------------------------------------------

def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int = TOURNAMENT_K
) -> np.ndarray:
    """Select one individual via tournament selection."""
    indices = np.random.randint(0, len(population), size=k)
    best_idx = indices[np.argmax(fitness[indices])]
    return population[best_idx].copy()


# ------------------------------------------------
# CROSSOVER: Simulated Binary Crossover (SBX)
# ------------------------------------------------

def _sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    eta: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """SBX crossover producing two offspring."""
    child1, child2 = p1.copy(), p2.copy()

    for i in range(NUM_GENES):
        if np.random.random() < 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                if p1[i] < p2[i]:
                    x1, x2 = p1[i], p2[i]
                else:
                    x1, x2 = p2[i], p1[i]

                rand = np.random.random()
                beta = 1.0 + (2.0 * (x1 - 0.01) / (x2 - x1 + 1e-14))
                alpha = 2.0 - beta ** (-(eta + 1.0))

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                child1[i] = 0.5 * ((1 + betaq) * x1 + (1 - betaq) * x2)
                child2[i] = 0.5 * ((1 - betaq) * x1 + (1 + betaq) * x2)

    return _normalize_chromosome(child1), _normalize_chromosome(child2)


# ------------------------------------------------
# MUTATION: Gaussian
# ------------------------------------------------

def _gaussian_mutate(chromo: np.ndarray, sigma: float = MUTATION_SIGMA) -> np.ndarray:
    """Apply Gaussian mutation and re-normalize."""
    noise = np.random.normal(0, sigma, size=NUM_GENES)
    mutated = chromo + noise
    return _normalize_chromosome(mutated)


# ------------------------------------------------
# MAIN GA ENGINE
# ------------------------------------------------

def evolve(
    fitness_fn: Callable[[np.ndarray], float],
    pop_size: int = POPULATION_SIZE,
    generations: int = NUM_GENERATIONS,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Run the genetic algorithm.

    Args:
        fitness_fn: Callable that takes a weight vector [4] and returns a fitness score.
        pop_size: Population size.
        generations: Number of generations.
        verbose: Print progress.

    Returns:
        (best_chromosome, best_fitness)
    """
    population = _init_population(pop_size)
    fitness = np.array([fitness_fn(ind) for ind in population])

    best_idx = np.argmax(fitness)
    best_chromo = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    if verbose:
        print(f"[GA] Gen 0 | Best fitness: {best_fitness:.4f} | Weights: {best_chromo}")

    for gen in range(1, generations + 1):
        new_population = []

        # Elitism: keep the best individual
        new_population.append(best_chromo.copy())

        while len(new_population) < pop_size:
            # Select parents
            parent1 = _tournament_select(population, fitness)
            parent2 = _tournament_select(population, fitness)

            # Crossover
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = _sbx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if np.random.random() < MUTATION_RATE:
                child1 = _gaussian_mutate(child1)
            if np.random.random() < MUTATION_RATE:
                child2 = _gaussian_mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = np.array(new_population[:pop_size])
        fitness = np.array([fitness_fn(ind) for ind in population])

        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_chromo = population[gen_best_idx].copy()

        if verbose and (gen % 5 == 0 or gen == 1):
            print(f"[GA] Gen {gen} | Best fitness: {best_fitness:.4f} | Weights: {np.round(best_chromo, 4)}")

    if verbose:
        print(f"\n[GA] Evolution complete!")
        print(f"[GA] Best weights: W_semantic={best_chromo[0]:.4f}, "
              f"W_cluster={best_chromo[1]:.4f}, "
              f"W_temporal={best_chromo[2]:.4f}, "
              f"W_actor={best_chromo[3]:.4f}")
        print(f"[GA] Best fitness: {best_fitness:.4f}")

    return best_chromo, best_fitness
