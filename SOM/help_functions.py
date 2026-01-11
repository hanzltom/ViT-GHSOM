import numpy as np

"""
Distance functions
"""
def euclidean_distance(a, b):
    diff = np.abs(a - b)
    return np.linalg.norm(diff, axis=2)

def manhattan_distance(a, b):
    diff = np.abs(a - b)
    return np.sum(diff, axis=2)

def chebyshev_distance(a, b):
    diff = np.abs(a - b)
    return np.max(diff, axis=2)

def generic_distance(a, b, k):
    diff = np.abs(a - b)
    return np.power(np.sum(np.power(diff, k), axis=2), 1.0 / k)


"""
Neighbourhood distance functions
"""
def gaussian_neighbourhood(grid_dists, sigma_t):
    return np.exp(- (grid_dists ** 2) / (2 * (sigma_t ** 2)))

def rectangular_neighbourhood(grid_dists, sigma_t):
    return (grid_dists <= sigma_t).astype(float)

def triangular_neighbourhood(grid_dists, sigma_t):
    return np.maximum(0.0, 1.0 - (grid_dists / sigma_t))

def cosine_down_to_zero_neighbourhood(grid_dists, sigma_t):
    influence = np.zeros_like(grid_dists)
    mask = grid_dists <= 2 * sigma_t

    influence[mask] = (np.cos((np.pi * grid_dists[mask]) / (2 * sigma_t)) + 1) / 2.0
    return influence


"""
Decay function
"""

def decay_exponential(initial_value, beta, t):
    return initial_value * (beta ** t)


def decay_power(initial_value, beta, t):
    return initial_value * (t ** beta)