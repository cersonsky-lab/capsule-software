import numpy as np
from tqdm.auto import tqdm

def make_histogram(data, sample_no, n_bins, scoring_func, label=None, ax=None):
    """Plot histogram of heterogeneity scores."""
    x, y, z = np.where(data == sample_no)
    if len(x) == 0:
        return

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

    pi_plot = pi.flatten()
    pi_plot = pi_plot[pi_plot != 0]
    
    if ax:
        ax.hist(pi_plot, bins=n_bins)
        ax.set_xlabel(label, fontsize=32)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_ylabel("Count", fontsize=32)
    
    return pi_plot
    

def plot_z_slices(data, sample_no, n_bins, scoring_func, ax=None, label=None):
    
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

    color = "tab:blue"
    if ax:
        ax.fill_between(
            np.arange(len(x_mean)),
            x_mean - x_var,
            x_mean + x_var,
            alpha=0.5,
            color=color,
            linewidth=0.5,
        )
        ax.plot(x_mean, color=color)
        ax.set_ylabel(label, color=color, fontsize=32)
        ax.tick_params(axis="y", labelcolor=color, labelsize=25)
        ax.tick_params(axis="x", labelsize=25)
        ax.set_xlabel("z", fontsize=32)

        color = "tab:red"
        ax2 = ax.twinx()
        ax2.plot(density, color=color)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=25)
        ax2.tick_params(axis="x", labelsize=25)
        ax2.set_ylabel("Density", color=color, fontsize=32)
    
    return x_mean, x_var, density

def plot_z_slices_test(data, sample_no, scoring_func, ax, label):
    """Plot z-slices of heterogeneity scores."""
    metrics = np.zeros(data.shape[0])
    density = np.zeros(data.shape[0])
    for zi in tqdm(range(data.shape[0])):
        metrics[zi] = scoring_func(data[zi, :, :], sample_no)
        density[zi] = np.sum(data[zi, :, :] == sample_no) / data.shape[1] / data.shape[2]
    
    color = "tab:blue"
    ax.plot(metrics, color=color)
    ax.set_ylabel(label, color=color, fontsize=30)
    ax.tick_params(axis="y", labelcolor=color, labelsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.set_xlabel("z", fontsize=32)
    
    ax2 = ax.twinx()
    color = "tab:red"
    ax2.plot(density, color=color, alpha=0.5)
    ax2.tick_params(axis="y", labelcolor=color, labelsize=25)
    ax2.set_ylabel("Density", color=color, fontsize=32)
    
    return metrics, density