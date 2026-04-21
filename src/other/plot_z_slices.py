import argparse

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("path")
parser.add_argument("n_bins", type=int)
parser.add_argument("metric")

args = parser.parse_args()

n_bins = args.n_bins
data = np.load(args.path)
cmap = plt.get_cmap("bone_r")
metric = args.metric


def calculate_poole_index(sub_array, sample_no):
    unique_labels = np.unique(sub_array)
    total_voxels = sub_array.size
    observed_species = len(
        np.where(sub_array.flatten() == sample_no)[0]
    )  # exclude the background (typically labeled as 0)
    poole_index = observed_species / total_voxels
    return poole_index


def calculate_lacey_index(sub_array, sample_no=None):

    is_species = (sub_array == sample_no).astype(int)

    if not np.any(is_species):
        return 0

    lacey_index = np.var(is_species) / np.mean(is_species)
    return lacey_index


def calculate_segregation_index(sub_array, sample_no=None):
    
    is_species = (sub_array == sample_no).astype(int)
    if not np.any(is_species):
        return 1
    segregation_index = 1 - (np.var(is_species) / np.mean(is_species))
    return segregation_index


def make_plot(sample_no, n_bins, scoring_func, label):
    x, y, z = np.where(data == sample_no)
    if len(x) == 0:
        print("bad")

    dl = (np.prod(data.shape) / n_bins) ** (1.0 / 3.0)
    n = np.array(data.shape // dl, dtype=int) + 1
    if len(x) == 0:
        print("bad")

    del x
    y_bins = np.linspace(0, data.shape[1], n[1], dtype=int)
    del y
    z_bins = np.linspace(0, data.shape[2], n[2], dtype=int)
    del z

    x_mean = np.zeros(data.shape[0])
    x_var = np.zeros(data.shape[0])
    density = np.zeros(data.shape[0])

    for xi in tqdm(range(data.shape[0])):
        metrics = np.zeros((len(z_bins) - 1) * (len(y_bins) - 1))
        for yi in range(len(y_bins) - 1):
            for zi in range(len(z_bins) - 1):
                metrics[zi * yi + yi] = scoring_func(
                    data[xi, y_bins[yi] : y_bins[yi + 1], z_bins[zi] : z_bins[zi + 1]],
                    sample_no,
                )

        x_mean[xi] = np.mean(metrics)
        x_var[xi] = np.var(metrics)
        slice = data[xi, :, :]
        # density[zi] = np.sum(slice == 2) / np.sum(slice != 0)
        density[xi] = np.sum(slice == 2) / slice.size

    fig, ax = plt.subplots()
    color = "tab:blue"
    ax.fill_between(
        np.arange(len(x_mean)),
        x_mean - x_var,
        x_mean + x_var,
        alpha=0.5,
        color=color,
        linewidth=0.5,
    )
    ax.plot(x_mean, color=color)
    ax.set_ylabel(label, color=color, fontsize=20)
    ax.tick_params(axis="y", labelcolor=color)
    ax.set_xlabel("z", fontsize=20)

    color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(density, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylabel("Density", color=color, fontsize=20)
    fig.tight_layout()
    plt.show()


if metric == "poole":
    scoring_func = calculate_poole_index
    label = "Poole Index"

elif metric == "lacey":
    scoring_func = calculate_lacey_index
    label = "Lacey Index"

elif metric == "segregation":
    scoring_func = calculate_segregation_index
    label = "Segregation Index"
else:
    raise ValueError("Invalid score")

make_plot(2, n_bins, scoring_func, label)
