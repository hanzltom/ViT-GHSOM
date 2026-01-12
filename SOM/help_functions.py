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


def generate_colormap(map, y, epoch_num):
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


def generate_vid(db, y):
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