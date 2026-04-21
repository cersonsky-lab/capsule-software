"""Code to remove the sample holder from the data based on smoothing splines"""

import argparse
import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_smoothing_spline
from skimage import data, filters, measure
from tqdm.auto import tqdm

SAMPLE_HOLDER_BUFFER = 50
parser = argparse.ArgumentParser()

parser.add_argument("path")
args = parser.parse_args()
path = args.path

if not os.path.exists(path.replace("nii", "npy")):
    DEFAULT_VAL = -1

    data = nib.load(path).get_fdata()[:, :, :]
    data -= data.min()

    def find_center(slice, plot=False):

        # Remove the background
        thresholds = filters.threshold_multiotsu(slice, classes=2)
        regions = np.digitize(slice, bins=thresholds)
        slice[regions == 0] = DEFAULT_VAL  # Remove background here

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(slice)

        binary_image = regions > 0  # API + sample holder
        if plot:
            ax[1].imshow(binary_image, vmin=0)
        labeled_image = measure.label(binary_image, 0)
        labeled_image[labeled_image == 0] = DEFAULT_VAL

        # Remove sample holders via connectivity-based labeling
        holders = np.where(labeled_image[:, labeled_image.shape[1] // 2] > 0)[0]  #
        labeled_image[
            labeled_image == labeled_image[holders[0], labeled_image.shape[1] // 2]
        ] = DEFAULT_VAL
        labeled_image[
            labeled_image == labeled_image[holders[-1], labeled_image.shape[1] // 2]
        ] = DEFAULT_VAL
        labeled_image[:SAMPLE_HOLDER_BUFFER, :] = DEFAULT_VAL
        labeled_image[-SAMPLE_HOLDER_BUFFER:, :] = DEFAULT_VAL
        labeled_image[:, -SAMPLE_HOLDER_BUFFER:] = DEFAULT_VAL
        labeled_image[labeled_image > 0] = 1
        if plot:
            ax[2].imshow(labeled_image, vmin=0)

        x, y = np.where(labeled_image > 0)
        if len(x) > 0 and len(np.where(labeled_image > 0)[0] > 100000):
            cx = np.mean([np.min(x), np.max(x)])
            cy = np.mean([np.min(y), np.max(y)])
            if plot:
                ax[2].axvline(cy)
                ax[2].axhline(cx)

            xs = np.where(labeled_image[int(cx)] > 0)[0]
            ys = np.where(labeled_image[int(cy)] > 0)[0]

            if len(xs) > 0 and len(ys) > 0:
                return cx, cy, max(xs) - min(xs), max(ys) - min(ys)
            else:
                return 0, 0, 0, 0

    def find_centers(slices, plot=False):
        centers = [[], []]
        d = [[], []]
        ies = []

        for i in tqdm(range(slices.shape[-1])):
            res = find_center(slices[:, :, i])
            if res is not None:
                centers[0].append(res[0])
                centers[1].append(res[1])

                d[0].append(res[2])
                d[1].append(res[3])
                ies.append(i)
        return np.array(ies), np.array(centers), np.array(d)

    if not os.path.exists(path.replace("nii", "npz")):
        xs, (cxs, cys), (dx, dy) = find_centers(data)
        np.savez(path.replace("nii", "npz"), xs=xs, cxs=cxs, cys=cys, dx=dx, dy=dy)
    else:
        npz_data = np.load(path.replace("nii", "npz"))
        xs = npz_data["xs"]
        cxs = npz_data["cxs"]
        cys = npz_data["cys"]
        dx = npz_data["dx"]
        dy = npz_data["dy"]

    xs = np.array([0, *xs, data.shape[-1] - 1])
    dx = np.array([0, *dx, 0])
    dy = np.array([0, *dy, 0])
    d = np.mean([dx, dy], axis=0)

    cutoffs = xs[np.where(d == 0)]
    min_cutoff = max(cutoffs[cutoffs < 100]) - 1
    max_cutoff = min(cutoffs[cutoffs > data.shape[-1] - 100]) + 1
    print(cutoffs, min_cutoff, max_cutoff)

    plt.plot(xs, d)
    plt.axvline(min_cutoff, c="k", linestyle="dashed")
    plt.axvline(max_cutoff, c="k", linestyle="dashed")

    i = np.where(
        np.logical_and(np.logical_and(xs > min_cutoff, xs < max_cutoff), d > 0)
    )[0]
    lam = d.shape[0] / 2.0
    f = make_smoothing_spline(xs[i], d[i], lam=lam)
    fx = f(xs[i])
    plt.plot(xs[i], fx)
    plt.show()

    i_trust = i[
        np.where(
            np.logical_or(
                np.logical_or(xs[i] == min(xs[i]), xs[i] == max(xs[i])),
                np.logical_and(d[i] >= fx * 0.999, d[i] <= fx / 0.999),
            )
        )
    ]
    x2 = xs.copy()[i_trust]
    dx2 = d.copy()[i_trust]

    plt.plot(xs, dx)
    plt.plot(x2, dx2)

    f2 = make_smoothing_spline(x2, dx2, lam=lam)
    fx2 = f2(x2)
    plt.plot(x2, fx2)
    plt.show()

    i_trust2 = np.where(
        np.logical_or(
            np.logical_or(x2 == min(x2), x2 == max(x2)),
            np.logical_and(dx2 >= fx2 * 0.999, dx2 <= fx2 / 0.999),
        )
    )

    x3 = x2[i_trust2].copy()
    dx3 = dx2[i_trust2].copy()

    plt.plot(x3, dx3)
    f2 = make_smoothing_spline(x3, dx3, lam=lam)
    fx2 = f2(x2)
    plt.plot(x2, fx2)
    # plt.show()

    cx = np.mean(cxs[cxs > 0])
    cy = np.mean(cys[cys > 0])

    r = np.zeros(data.shape[-1])
    x2s = np.arange(xs[i].min(), xs[i].max(), dtype=int)
    print(x2s)
    r[x2s] = f2(x2s)

    plt.plot(np.arange(data.shape[-1]), r)
    plt.show()

    def remove_items(slice, rr):

        thresholds = filters.threshold_multiotsu(slice, classes=3)
        regions = np.digitize(slice, bins=thresholds)
        # slice[regions == 0] = DEFAULT_VAL

        x, y = np.indices(slice.shape)
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = np.array(distances > rr / 2.0).reshape(slice.shape)
        slice[mask] = DEFAULT_VAL
        return slice

    masked_slices = np.array(
        [remove_items(data[:, :, i], r[i]) for i in tqdm(range(data.shape[-1]))]
    )
    masked_slices.shape

    # thresholds = filters.threshold_multiotsu(masked_slices, classes=3)
    # regions = np.digitize(masked_slices, bins=thresholds)
    np.save(path.replace("nii", "npy"), masked_slices)
    np.save(path.replace(".nii", "r.npy"), r)
    # os.remove(path.replace("nii", "npz"))
else:
    regions = np.load(path.replace("nii", "npy"))

plt.imshow(masked_slices[10])
plt.show()
plt.imshow(masked_slices[:, 375])
plt.show()
plt.imshow(masked_slices[:, np.random.choice(masked_slices.shape[1])])
plt.show()
