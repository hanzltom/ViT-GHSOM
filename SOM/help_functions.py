# Filename: help_functions.py
# Author: Tomas Hanzlik <hanzlto3@fit.cvut.cz>
# Created: 2026-04-22
# Description: Help functions for the training of the standard SOM algorithm.

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation, colors
from IPython.display import HTML
from typing import Callable

"""
Distance functions
"""

def euclidean_distance(a: np.ndarray, b: np.ndarray, axis: int | None) -> np.ndarray:
    """
    Euclidean distance between two vectors.
    :param a: Vector a
    :param b: Vector b
    :param axis: Axis where distance is computed
    :return: Distance
    """
    diff = np.abs(a - b)
    return np.linalg.norm(diff, axis=axis)

def manhattan_distance(a: np.ndarray, b: np.ndarray, axis: int | None) -> np.ndarray:
    """
    Manhattan distance between two vectors.
    :param a: Vector a
    :param b: Vector b
    :param axis: Axis where distance is computed
    :return: Distance
    """
    diff = np.abs(a - b)
    return np.sum(diff, axis=axis)

def chebyshev_distance(a: np.ndarray, b: np.ndarray, axis: int | None) -> np.ndarray:
    """
    Chebyshev distance between two vectors.
    :param a: Vector a
    :param b: Vector b
    :param axis: Axis where distance is computed
    :return: Distance
    """
    diff = np.abs(a - b)
    return np.max(diff, axis=axis)

def generic_distance(a: np.ndarray, b: np.ndarray, axis: int, k: int | None) -> np.ndarray:
    """
    Generic distance from Minskowski distance functions
    :param a: Vector a
    :param b: Vector b
    :param axis: Axis where distance is computed
    :param k: k for the formula
    :return: Distance
    """
    diff = np.abs(a - b)
    return np.power(np.sum(np.power(diff, k), axis=axis), 1.0 / k)


"""
Neighbourhood distance functions
"""
def gaussian_neighbourhood(grid_dists: np.ndarray, sigma_t: float) -> np.ndarray:
    """
    Gaussian neighbourhood influence function
    :param grid_dists: Distance from the BMU to other neurons
    :param sigma_t: Sigma at given time
    :return: Gaussian influence
    """
    return np.exp(- (grid_dists ** 2) / (2 * (sigma_t ** 2)))

def rectangular_neighbourhood(grid_dists: np.ndarray, sigma_t: float) -> np.ndarray:
    """
    Rectangular neighbourhood influence function
    :param grid_dists: Distance from the BMU to other neurons
    :param sigma_t: Sigma at given time
    :return: Rectangular influence
    """
    return (grid_dists <= sigma_t).astype(float)

def triangular_neighbourhood(grid_dists: np.ndarray, sigma_t: float) -> np.ndarray:
    """
    Triangular neighbourhood influence function
    :param grid_dists: Distance from the BMU to other neurons
    :param sigma_t: Sigma at given time
    :return: Triangular influence
    """
    return np.maximum(0.0, 1.0 - (grid_dists / sigma_t))

def cosine_down_to_zero_neighbourhood(grid_dists: np.ndarray, sigma_t: float) -> np.ndarray:
    """
    Cosine down to zero neighbourhood influence function
    :param grid_dists: Distance from the BMU to other neurons
    :param sigma_t: Sigma at given time
    :return: Cosine influence
    """
    influence = np.zeros_like(grid_dists)
    mask = grid_dists <= 2 * sigma_t

    influence[mask] = (np.cos((np.pi * grid_dists[mask]) / (2 * sigma_t)) + 1) / 2.0
    return influence


"""
Decay function
"""

def decay_exponential(initial_value: float, beta: float, t: int) -> float:
    """
    Decay exponential function
    :param initial_value: Initial value
    :param beta: Beta value, must satisfy: 0 < beta < 1
    :param t: Current time
    :return: Decayed initial value
    """
    return initial_value * (beta ** t)


def decay_power(initial_value: float, beta: float, t: int) -> float:
    """
    Decay power function
    :param initial_value: Initial value
    :param beta: Beta value, must satisfy: beta < 0
    :param t: Current time
    :return: Decayed initial value
    """
    return initial_value * (t ** beta)


"""
Visualization
"""


def generate_label_matrix(weight_matrix: np.ndarray,
                          data: np.ndarray,
                          labels: np.ndarray,
                          map_rows: int,
                          map_cols: int,
                          calculate_distance_func: Callable[[np.ndarray, np.ndarray, int], float]
                          ) -> np.ndarray:
    """
    Function to generate a matrix where the label of each node represents the major class
    :param weight_matrix: SOM weight matrix
    :param data: Dataset
    :param labels: Target labels
    :param map_rows: Number of rows in the map
    :param map_cols: Number of columns in the map
    :param calculate_distance_func: Distance function
    :return: Matrix with labels
    """
    # https://medium.com/data-science/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    map = np.empty(shape=(map_rows, map_cols), dtype=object)

    for row in range(map_rows):
        for col in range(map_cols):
            map[row][col] = []

    for i, sample in enumerate(data):
        dists = calculate_distance_func(weight_matrix, sample, 2)
        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)

        map[bmu_idx[0]][bmu_idx[1]].append(labels[i])

    for row in range(map_rows):
        for col in range(map_cols):
            label_list = map[row][col]
            if len(label_list) == 0:
                label = np.nan
            else:
                # get the label of the majority class
                label = max(label_list, key=label_list.count)
            map[row][col] = label

    return map.astype(float)


def generate_label_matrix_db(weights_db: dict[int: np.ndarray],
                             data: np.ndarray,
                             labels: np.ndarray,
                             map_rows: int,
                             map_cols: int,
                             calculate_distance_func: Callable[[np.ndarray, np.ndarray, int], float]
                             ) -> dict[int: np.ndarray]:
    """
    Function to generate label matrix for each epoch in the SOM weights database
    :param weights_db: SOM weights database
    :param data: Dataset
    :param labels: Target labels
    :param map_rows: Number of rows in the map
    :param map_cols: Number of columns in the map
    :param calculate_distance_func: Distance function
    :return: Dictionary of label matrix for each epoch
    """
    label_matrix_db = {}
    for epoch, weight_matrix in weights_db.items():
        label_matrix_db[epoch] = generate_label_matrix(weight_matrix, data, labels, map_rows, map_cols, calculate_distance_func)

    return label_matrix_db


def visualize_label_matrix(som: "SOM", y: np.ndarray, epoch_num: int, name : str = None):
    """
    Function to visualize label matrix
    :param som: SOM class
    :param y: Target labels
    :param epoch_num: Epoch for which we want the label matrix
    :param name: Name of the dataset for title
    """

    map = som.label_matrix_db[epoch_num]
    y_unique = np.unique(y)

    # define colors
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan']

    if len(y_unique) > len(color_options):
        raise ValueError("Too many target variables, not enough color choices")
    cmap = colors.ListedColormap(color_options[:len(y_unique)])
    cmap.set_bad(color='lightgrey')

    # create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map, cmap=cmap)
    if name:
        ax.set_title(f"SOM grid with majority class, Epoch: {epoch_num}, {name} dataset")
    else:
        ax.set_title(f"SOM grid with majority class, Epoch: {epoch_num}")

    rows, cols = map.shape
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # create patches for the legend
    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='lightgrey', label='Empty'))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def generate_label_matrix_vid(som: "SOM", y: np.ndarray):
    """
    Function to animate change of label matrix during training
    :param som:
    :param y:
    :return: HTML video
    """
    db = som.label_matrix_db
    epochs = sorted(db.keys())
    y_unique = np.unique(y)

    fig, ax = plt.subplots(figsize=(8, 8))
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan']
    num_repeats = math.ceil(len(y_unique) / len(color_options))
    extended_colors = (color_options * num_repeats)[:len(y_unique)]
    cmap = colors.ListedColormap(extended_colors)

    im = ax.imshow(db[1], cmap=cmap)

    rows, cols = som.map_rows, som.map_cols
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='tab:grey', label='Empty'))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    def update(frame_idx):
        epoch = epochs[frame_idx]
        map = db[epoch]

        im.set_data(map)
        ax.set_title(f"Epoch: {epoch}")
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=200,
        blit=True
    )

    plt.close()
    return HTML(anim.to_jshtml())


def generate_u_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    """
    Function to generate U-Matrix
    :param weight_matrix: Weights matrix
    :return: U-Matrix
    """
    m, n, dim = weight_matrix.shape
    u_matrix = np.zeros((m, n))

    # for each neuron, calculate the mean distance to its neighbours in rectangular grid
    for r in range(m):
        for c in range(n):
            distances = []
            current_neuron = weight_matrix[r, c]
            coords_neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for rn, cn in coords_neighbours:
                if 0 <= cn < n and 0 <= rn < m:
                    neighbor_neuron = weight_matrix[rn, cn]

                    # calculate Euclidean distance
                    dist = np.linalg.norm(current_neuron - neighbor_neuron)
                    distances.append(dist)

            u_matrix[r, c] = np.mean(distances)

    return u_matrix


def generate_u_matrix_db(weights_db: dict[int: np.ndarray]) -> dict[int: np.ndarray]:
    """
    Function to generate U-Matrix database for each epoch
    :param weights_db: Database of weights for each epoch
    :return: Database of U-Matrices
    """
    u_matrix_db = {}
    for epoch, weight_matrix in weights_db.items():
        u_matrix_db[epoch] = generate_u_matrix(weight_matrix)

    return u_matrix_db


def visualize_u_matrix(som: "SOM", epoch_num: int, name : str = None):
    """
    Function to visualize U-Matrix
    :param som: SOM class
    :param epoch_num: Epoch for which we want the U-Matrix
    :param name: Name of the dataset for title
    """
    matrix = som.u_matrix_db[epoch_num]
    ig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(matrix, cmap='plasma')

    if name:
        ax.set_title(f"U-Matrix, Epoch: {epoch_num}, {name} dataset")
    else:
        ax.set_title(f"U-Matrix, Epoch: {epoch_num}")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Distance to Neighbours')

    rows, cols = matrix.shape
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    plt.tight_layout()
    plt.show()


def generate_u_matrix_vid(som):
    """
    Function to animate change of U-Matrix during training
    :param som: SOM class
    :return: HTML video
    """
    db = som.u_matrix_db
    epochs = sorted(db.keys())

    fig, ax = plt.subplots(figsize=(8, 8))
    values = [db[e] for e in epochs]

    im = ax.imshow(db[1], cmap='plasma', vmin=np.min(values), vmax=np.max(values))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Distance to Neighbours')

    plt.tight_layout()

    def update(frame_idx):
        epoch = epochs[frame_idx]
        map = db[epoch]

        im.set_data(map)
        ax.set_title(f"Epoch: {epoch}")
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=200,
        blit=True
    )

    plt.close()
    return HTML(anim.to_jshtml())


def generate_extended_u_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    """
    Function to generate Extended U-Matrix
    :param weight_matrix: Weights matrix
    :return: Extended U-Matrix
    """
    m, n, dim = weight_matrix.shape

    # Size of extended U-Matrix (2m-1)*(2n-1)
    ext_m = 2 * m - 1
    ext_n = 2 * n - 1
    u_matrix_extended = np.zeros((ext_m, ext_n))

    # Horizontal distances
    for r in range(m):
        for c in range(n - 1):
            current_node = weight_matrix[r, c]
            right_neighbour = weight_matrix[r, c + 1]

            dist = np.linalg.norm(current_node - right_neighbour)
            u_matrix_extended[2 * r, 2 * c + 1] = dist

    # Vertical distances
    for r in range(m - 1):
        for c in range(n):
            current_node = weight_matrix[r, c]
            bottom_neighbour = weight_matrix[r + 1, c]

            dist = np.linalg.norm(current_node - bottom_neighbour)
            u_matrix_extended[2 * r + 1, 2 * c] = dist

    # Centers among distances - odd rows, odd columns - average over neighbours
    for r in range(1, ext_m, 2):
        for c in range(1, ext_n, 2):
            neighbours = [u_matrix_extended[r - 1, c], u_matrix_extended[r + 1, c], u_matrix_extended[r, c - 1],
                         u_matrix_extended[r, c + 1]]
            u_matrix_extended[r, c] = np.mean(neighbours)

    # Neuron positions - average over neighbours
    for r in range(0, ext_m, 2):
        for c in range(0, ext_n, 2):
            distances = []
            if r > 0: distances.append(u_matrix_extended[r - 1, c]) #top
            if r < ext_m - 1: distances.append(u_matrix_extended[r + 1, c]) #bottom
            if c > 0: distances.append(u_matrix_extended[r, c - 1]) #left
            if c < ext_n - 1: distances.append(u_matrix_extended[r, c + 1]) #right

            u_matrix_extended[r,c] = np.mean(distances)

    return u_matrix_extended

def generate_u_matrix_extended_db(weights_db: dict[int: np.ndarray]) -> dict[int: np.ndarray]:
    """
    Function to generate Extended U-Matrix database for each epoch
    :param weights_db: Database of weights for each epoch
    :return: Database of Extended U-Matrices
    """
    u_matrix_extended_db = {}
    for epoch, weight_matrix in weights_db.items():
        u_matrix_extended_db[epoch] = generate_extended_u_matrix(weight_matrix)

    return u_matrix_extended_db

def visualize_u_matrix_extended(som: "SOM", epoch_num: int, name : str = None):
    """
    Function to visualize U-Matrix
    :param som: SOM class
    :param epoch_num: Epoch for which we want the U-Matrix
    :param name: Name of the dataset for title
    """
    matrix = som.u_matrix_extended_db[epoch_num]
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(matrix, cmap='plasma')

    m, n = matrix.shape
    ax.set_xticks(np.arange(0, n, 2))
    ax.set_yticks(np.arange(0, m, 2))
    ax.set_xticklabels(np.arange(len(np.arange(0, n, 2))))
    ax.set_yticklabels(np.arange(len(np.arange(0, m, 2))))

    neuron_x_coords = []
    neuron_y_coords = []

    for r in range(0, m, 2):
        for c in range(0, n, 2):
            neuron_y_coords.append(r)
            neuron_x_coords.append(c)

    ax.scatter(neuron_x_coords, neuron_y_coords, s=30, c='yellow', edgecolors='black', label='Neuron Position')

    if name:
        ax.set_title(f"U-Matrix Extended, Epoch: {epoch_num}, {name} dataset")
    else:
        ax.set_title(f"U-Matrix Extended, Epoch: {epoch_num}")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance')
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))

    plt.tight_layout()
    plt.show()

def generate_u_matrix_extended_vid(som):
    """
        Function to animate change of Extended U-Matrix during training
        :param som: SOM class
        :return: HTML video
        """
    db = som.u_matrix_extended_db
    epochs = sorted(db.keys())

    fig, ax = plt.subplots(figsize=(10, 10))
    values = [db[e] for e in epochs]

    im = ax.imshow(db[1], cmap='plasma', vmin=np.min(values), vmax=np.max(values))
    m, n = values[0].shape

    neuron_x_coords = []
    neuron_y_coords = []

    for r in range(0, m, 2):
        for c in range(0, n, 2):
            neuron_y_coords.append(r)
            neuron_x_coords.append(c)

    ax.scatter(neuron_x_coords, neuron_y_coords, s=30, c='yellow', edgecolors='black', label='Neuron Position')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance')
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()

    def update(frame_idx):
        epoch = epochs[frame_idx]
        map = db[epoch]

        im.set_data(map)
        ax.set_title(f"Epoch: {epoch}")
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=200,
        blit=True
    )

    plt.close()
    return HTML(anim.to_jshtml())