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

def generic_distance(a: np.ndarray, b: np.ndarray, axis: int | None, k: int) -> np.ndarray:
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
                          current_row_num: int,
                          current_col_num: int,
                          calculate_distance_func: Callable[[np.ndarray, np.ndarray, int], float]
                          ) -> np.ndarray:
    """
    Function to generate a matrix where the label of each node represents the major class
    :param weight_matrix: SOM weight matrix
    :param data: Dataset
    :param labels: Target labels
    :param current_row_num: Number of rows in the map
    :param current_col_num: Number of columns in the map
    :param calculate_distance_func: Distance function
    :return: Matrix with labels
    """
    # https://medium.com/data-science/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    map = np.empty(shape=(current_row_num, current_col_num), dtype=object)

    for row in range(current_row_num):
        for col in range(current_col_num):
            map[row][col] = []

    for i, sample in enumerate(data):
        dists = calculate_distance_func(weight_matrix, sample, 2)
        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)

        map[bmu_idx[0]][bmu_idx[1]].append(labels[i])

    for row in range(current_row_num):
        for col in range(current_col_num):
            label_list = map[row][col]
            if len(label_list) == 0:
                label = np.nan
            else:
                label = max(label_list, key=label_list.count)
            map[row][col] = label

    return map.astype(float)


def generate_label_matrix_db(gsom_db: dict[str: np.ndarray],
                             data: np.ndarray,
                             labels: np.ndarray,
                             current_row_num: int,
                             current_col_num: int,
                             calculate_distance_func: Callable[[np.ndarray, np.ndarray, int], float]
                             ) -> dict[int: np.ndarray]:
    """
    Function to generate label matrix for each epoch in the SOM weights database
    :param gsom_db: SOM weights database
    :param data: Dataset
    :param labels: Target labels
    :param current_row_num: Number of rows in the map
    :param current_col_num: Number of columns in the map
    :param calculate_distance_func: Distance function
    :return: Dictionary of label matrix for each epoch
    """
    label_matrix_db = {}
    for id_gsom, gsom in gsom_db.items():
        label_matrix_db[id_gsom] = generate_label_matrix(gsom.weights, data, labels, current_row_num, current_col_num, calculate_distance_func)

    return label_matrix_db


def visualize_label_matrix(ghsom: "GHSOM", y: np.ndarray, gsom_id: str):
    """
    Function to visualize individual label matrix of given gsom
    :param ghsom: GHSOM instance
    :param y: Target labels
    :param gsom_id: ID of GSOM to visualize its label matrix
    """
    map = ghsom.label_matrix_db[gsom_id]
    y_unique = np.unique(y)

    # define colors
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan']
    cmap = colors.ListedColormap(color_options[:len(y_unique)])
    cmap.set_bad(color='lightgrey')

    # create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map, cmap=cmap)
    ax.set_title(f"GSOM id: {gsom_id}")

    # create patches for the legend
    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='lightgrey', label='Empty'))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def draw_recursive(ax: plt.axes,
                   ghsom: "GHSOM",
                   map_id: str,
                   x_start: float,
                   y_start: float,
                   width: float,
                   height: float,
                   label_map: dict[str: str],
                   color_map:  dict[str, str],
                   depth: int):
    """
    Recursive function to plot each individual GSOM label matrix
    :param ax: Matplotlib Axes
    :param ghsom: GHSOM instance
    :param map_id: ID of current GSOM unit
    :param x_start: Position on x-axis where to start drawing
    :param y_start: Position on y-axis where to start drawing
    :param width: Total width for current GSOM map
    :param height: Total height for current GSOM map
    :param label_map: Mapping of labels on each GSOM units
    :param color_map: Color encoding for each target label
    :param depth: Current depth of recursion
    """
    current_gsom = ghsom.gsom_db[map_id]

    # dimensions of single neuron
    unit_w = width / current_gsom.current_col_num
    unit_h = height / current_gsom.current_row_num
    linewidth = max(0.5, 3.0 - depth * 0.8)

    # iterate over neurons in grid convention (row 0 at top)
    for r in range(current_gsom.current_row_num):
        for c in range(current_gsom.current_col_num):
            unit_id = f"{map_id}_{r}-{c}"

            # calculate bottom left corner of rect
            # (current_gsom.current_row_num - 1 - r) flips the row index, otherwise it is upside down
            unit_x = x_start + c * unit_w
            unit_y = y_start + (current_gsom.current_row_num - 1 - r) * unit_h

            # Recursion, check for children GSOM
            if unit_id in ghsom.gsom_db:
                draw_recursive(ax, ghsom, unit_id, unit_x, unit_y, unit_w, unit_h, label_map, color_map, depth + 1)
            else:
                # Leaf node - add Patch
                label = label_map.get(unit_id, "Empty")
                color = color_map[label]

                rect = mpatches.Rectangle((unit_x, unit_y), unit_w, unit_h,
                                          facecolor=color, edgecolor='white', linewidth=0.5, zorder=0)
                ax.add_patch(rect)

    # Draws edges
    outline = mpatches.Rectangle((x_start, y_start), width, height,
                                 fill=False, edgecolor='black', linewidth=linewidth + 1, zorder=depth + 10)
    ax.add_patch(outline)


def plot_ghsom(ghsom_instance: "GHSOM", X: np.ndarray, y: np.ndarray):
    """
    Function which plots GHSOM with all GSOM instances and their label matrices
    :param ghsom_instance: GHSOM instance
    :param X: Dataset
    :param y: Target labels
    """

    hierarchy_label_map = ghsom_instance.get_labels(X, y)

    label_names, y_int = np.unique(y, return_inverse=True)
    unique_labels = [l for l in label_names if l in hierarchy_label_map.values()]
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    label_to_color = {label: color_options[i % len(color_options)] for i, label in enumerate(unique_labels)}
    label_to_color["Empty"] = "tab:grey"

    fig, ax = plt.subplots(figsize=(12, 12))

    # canvas bounds
    canvas_x, canvas_y = 0.0, 0.0
    canvas_w, canvas_h = 1.0, 1.0
    max_depth = max([key.count('_') for key in ghsom_instance.gsom_db.keys()]) + 1

    draw_recursive(ax, ghsom_instance, "1", canvas_x, canvas_y, canvas_w, canvas_h, hierarchy_label_map, label_to_color,
                   depth=0)

    ax.axis('off')
    ax.set_title(f"GHSOM Structure (Max Depth: {max_depth})")
    patches = [mpatches.Patch(color=label_to_color[l], label=l) for l in unique_labels]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()



