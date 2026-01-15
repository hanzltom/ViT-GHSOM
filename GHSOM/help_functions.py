import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation, colors
from IPython.display import HTML

"""
Distance functions
"""

def euclidean_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.linalg.norm(diff, axis=axis)

def manhattan_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.sum(diff, axis=axis)

def chebyshev_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.max(diff, axis=axis)

def generic_distance(a, b, axis, k):
    diff = np.abs(a - b)
    return np.power(np.sum(np.power(diff, k), axis=axis), 1.0 / k)

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


"""
Visualization
"""


def generate_label_matrix(gsom, weight_matrix, data, labels):
    # https://medium.com/data-science/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    map = np.empty(shape=(gsom.current_row_num, gsom.current_col_num), dtype=object)

    for row in range(gsom.current_row_num):
        for col in range(gsom.current_col_num):
            map[row][col] = []

    for i, sample in enumerate(data):
        dists = gsom.calculate_distance_func(weight_matrix, sample, 2)
        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)

        map[bmu_idx[0]][bmu_idx[1]].append(labels[i])

    for row in range(gsom.current_row_num):
        for col in range(gsom.current_col_num):
            label_list = map[row][col]
            if len(label_list) == 0:
                label = np.nan
            else:
                label = max(label_list, key=label_list.count)
            map[row][col] = label

    return map.astype(float)


def generate_label_matrix_db(model, data, labels):
    label_matrix_db = {}
    for id_gsom, gsom in model.gsom_db.items():
        label_matrix_db[id_gsom] = generate_label_matrix(gsom, gsom.weights, data, labels)

    return label_matrix_db


def visualize_label_matrix(map, y, gsom_id):
    y_unique = np.unique(y)
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
    cmap = colors.ListedColormap(color_options[:len(y_unique)])
    cmap.set_bad(color='lightgrey')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map, cmap=cmap)

    m, n = map.shape

    neuron_x_coords = []
    neuron_y_coords = []

    for r in range(0, m):
        for c in range(0, n):
            neuron_y_coords.append(r)
            neuron_x_coords.append(c)

    ax.scatter(neuron_x_coords, neuron_y_coords, s=30, c='yellow', edgecolors='black', label='Neuron Position')

    ax.set_title(f"GSOM id: {gsom_id}")
    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='lightgrey', label='Empty'))
    ax.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.show()


def draw_recursive(ax, ghsom, map_id, x_start, y_start, width, height, label_map, color_map, depth):
    current_gsom = ghsom.gsom_db[map_id]

    # dimensions of single neuron
    unit_w = width / current_gsom.current_col_num
    unit_h = height / current_gsom.current_row_num
    linewidth = max(0.5, 3.0 - depth * 0.8)

    # Iterate over neurons in grid convention (row 0 at top)
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
                # Leaf node
                label = label_map.get(unit_id, "Empty")
                color = color_map[label]

                rect = mpatches.Rectangle((unit_x, unit_y), unit_w, unit_h,
                                          facecolor=color, edgecolor='white', linewidth=0.5, zorder=0)
                ax.add_patch(rect)

    # Draws edges
    outline = mpatches.Rectangle((x_start, y_start), width, height,
                                 fill=False, edgecolor='black', linewidth=linewidth + 1, zorder=depth + 10)
    ax.add_patch(outline)


def plot_ghsom(ghsom_instance, X_full, y_full, label_names):
    hierarchy_label_map = ghsom_instance.get_labels(X_full, y_full, label_names)

    unique_labels = [l for l in label_names if l in hierarchy_label_map.values()]
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink']
    label_to_color = {label: color_options[i % len(color_options)] for i, label in enumerate(unique_labels)}
    label_to_color["Empty"] = "tab:grey"

    fig, ax = plt.subplots(figsize=(12, 12))

    # canvas bounds
    canvas_x, canvas_y = 0.0, 0.0
    canvas_w, canvas_h = 1.0, 1.0

    draw_recursive(ax, ghsom_instance, "1", canvas_x, canvas_y, canvas_w, canvas_h, hierarchy_label_map, label_to_color,
                   depth=0)

    ax.axis('off')
    ax.set_title(f"GHSOM Structure")
    patches = [mpatches.Patch(color=label_to_color[l], label=l) for l in unique_labels]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()



