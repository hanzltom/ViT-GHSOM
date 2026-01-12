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


def generate_label_matrix(model, weight_matrix, data, labels):
    # https://medium.com/data-science/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    map = np.empty(shape=(model.map_width, model.map_height), dtype=object)

    for row in range(model.map_width):
        for col in range(model.map_height):
            map[row][col] = []

    for i, sample in enumerate(data):
        dists = model.calculate_distance_func(weight_matrix, sample, 2)
        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)

        map[bmu_idx[0]][bmu_idx[1]].append(labels[i])

    for row in range(model.map_width):
        for col in range(model.map_height):
            label_list = map[row][col]
            if len(label_list) == 0:
                label = np.nan
            else:
                label = max(label_list, key=label_list.count)
            map[row][col] = label

    return map.astype(float)


def generate_label_matrix_db(model, data, labels):
    label_matrix_db = {}
    for epoch, weight_matrix in model.weights_db.items():
        label_matrix_db[epoch] = generate_label_matrix(model, weight_matrix, data, labels)

    return label_matrix_db


def visualize_label_matrix(map, y, epoch_num):
    y_unique = np.unique(y)
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
    cmap = colors.ListedColormap(color_options[:len(y_unique)])
    cmap.set_bad(color='lightgrey')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map, cmap=cmap)
    ax.set_title(f"Epoch: {epoch_num}")
    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='lightgrey', label='Empty'))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def generate_label_matrix_vid(db, y):
    epochs = sorted(db.keys())
    y_unique = np.unique(y)

    fig, ax = plt.subplots(figsize=(8, 8))
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
    cmap = colors.ListedColormap(color_options[:len(y_unique)])

    im = ax.imshow(db[0], cmap=cmap)

    patches = [mpatches.Patch(color=color_options[i], label=label) for i, label in enumerate(y_unique)]
    patches.append(mpatches.Patch(color='lightgrey', label='Empty'))
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


def generate_u_matrix(weight_matrix):
    width, height, dim = weight_matrix.shape
    u_matrix = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            distances = []
            current_neuron = weight_matrix[x, y]
            coords_neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for nx, ny in coords_neighbours:
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_neuron = weight_matrix[nx, ny]

                    dist = np.linalg.norm(current_neuron - neighbor_neuron)
                    distances.append(dist)

            u_matrix[x, y] = np.mean(distances)

    return u_matrix


def generate_u_matrix_db(model):
    u_matrix_db = {}
    for epoch, weight_matrix in model.weights_db.items():
        u_matrix_db[epoch] = generate_u_matrix(weight_matrix)

    return u_matrix_db


def visualize_u_matrix(matrix, epoch_num):
    ig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(matrix, cmap='plasma')
    ax.set_title(f"U-Matrix (Epoch: {epoch_num})")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Distance to Neighbours')

    plt.tight_layout()
    plt.show()

def generate_u_matrix_vid(db):
    epochs = sorted(db.keys())

    fig, ax = plt.subplots(figsize=(8, 8))
    values = [db[e] for e in epochs]

    im = ax.imshow(db[0], cmap='plasma', vmin=np.min(values), vmax=np.max(values))
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