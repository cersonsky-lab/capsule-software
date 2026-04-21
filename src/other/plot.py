import argparse

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("path")
parser.add_argument("n_bins", type=int)
parser.add_argument("n_levels", type=int)

args = parser.parse_args()

n_bins = args.n_bins
data = np.load(args.path)
cmap = plt.get_cmap("bone_r")


def calculate_poole_index(sub_array, sample_no):
    unique_labels = np.unique(sub_array)
    total_voxels = sub_array.size
    observed_species = len(
        np.where(sub_array.flatten() == sample_no)[0]
    )  # exclude the background (typically labeled as 0)
    poole_index = observed_species / total_voxels
    return poole_index


def calculate_lacey_index(sub_array, sample_no=None):
    total_voxels = sub_array.size
    unique_labels, counts = np.unique(sub_array, return_counts=True)
    counts = counts[1:]  # exclude the background (typically labeled as 0)

    lacey_index = np.var(counts) / np.mean(counts)
    return lacey_index


def calculate_segregation_index(sub_array, sample_no=None):
    unique_labels, counts = np.unique(sub_array, return_counts=True)
    counts = counts[1:]  # exclude the background (typically labeled as 0)

    segregation_index = 1 - (np.var(counts) / np.mean(counts))
    return segregation_index


def make_plot(sample_no, n_bins, scoring_func):
    x, y, z = np.where(data == sample_no)

    dl = (np.prod(data.shape) / n_bins) ** (1.0 / 3.0)
    n = np.array(data.shape // dl, dtype=int) + 1
    if len(x) == 0:
        return

    x_bins = np.linspace(0, data.shape[0], n[0], dtype=int)
    del x
    y_bins = np.linspace(0, data.shape[1], n[1], dtype=int)
    del y
    z_bins = np.linspace(0, data.shape[2], n[2], dtype=int)
    del z

    pi = np.zeros(n)

    for xi in tqdm(range(len(x_bins) - 1)):
        for yi in range(len(y_bins) - 1):
            for zi in range(len(z_bins) - 1):
                pi[xi, yi, zi] = scoring_func(
                    data[
                        x_bins[xi] : x_bins[xi + 1],
                        y_bins[yi] : y_bins[yi + 1],
                        z_bins[zi] : z_bins[zi + 1],
                    ],
                    sample_no,
                )
    print(pi.min(), pi.max())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mpl.colors.Normalize(vmin=pi.min(), vmax=pi.max())
    my_cmap = lambda i: cmap(norm(i))
    iso_vals = np.round(np.linspace(0, 0.5, args.n_levels + 1), 4)[1:]
    iso_vals = iso_vals[iso_vals > pi.min()]
    iso_vals = iso_vals[iso_vals < pi.max()]
    for iso_val in iso_vals:
        # try:
        verts, faces, _, _ = measure.marching_cubes(pi, iso_val, spacing=n)
        if len(verts) > 50:
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                faces,
                verts[:, 2],
                lw=1,
                alpha=iso_val**2.0,
                color=my_cmap(iso_val),
                label=iso_val,
            )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_zticks([])
    plt.gca().set_title(args.path + ":\nSample {}".format(sample_no))
    plt.legend()
    plt.show()


make_plot(2, args.n_bins, calculate_poole_index)
