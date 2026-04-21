import argparse

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from tqdm.auto import tqdm
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("n_bins", type=int)
parser.add_argument("output_dir")
parser.add_argument("--tune", action="store_true")

args = parser.parse_args()

n_bins = args.n_bins
data = np.load(args.path)
tune = args.tune
output_dir = args.output_dir

data = data == 2 # Only care about API for this part

PAD = 100

distance_map = np.zeros(data.shape)

bins = np.linspace(0, data.shape[0], n_bins+1, dtype=int)
for xi in tqdm(range(len(bins) - 1)):
# 1. Define the CORE region for this bin (where we will write data)
            core_x_start, core_x_end = bins[xi], bins[xi+1]
            
            # 2. Define the PADDED region to extract for processing
            padded_x_start = max(core_x_start - PAD, 0)
            padded_x_end = min(core_x_end + PAD, data.shape[0])

            # Extract the padded sub-array from the source data
            sub_array = data[padded_x_start:padded_x_end, :, :]

            # 3. Run the distance transform on the padded sub-array
            distance_result = distance_transform_edt(sub_array)

            # 4. Calculate the slice to extract the VALID result from the processed array
            # This corresponds to the location of the core region within the padded region.
            res_x_start = core_x_start - padded_x_start
            res_x_end = res_x_start + (core_x_end - core_x_start)

            # 5. Copy the valid core data into the final output map
            core_slice_out = np.s_[core_x_start:core_x_end, :, :]
            core_slice_in = np.s_[res_x_start:res_x_end, :, :]

            distance_map[core_slice_out] = distance_result[core_slice_in]


if tune:
    
    min_distances = np.linspace(5, 30, 20).astype(int)
    n_crystals = []
    for min_distance in tqdm(min_distances):
        local_maxima_coords = peak_local_max(
            distance_map, 
            min_distance=min_distance,
            labels=(data)  # Use the original boolean mask to exclude peaks in the background
        )
        print(f"Min Distance: {min_distance}, Found Peaks: {len(local_maxima_coords)}")
        n_crystals.append(len(local_maxima_coords))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(min_distances, n_crystals)
    plt.xlabel("Minimum Distance Between Markers (voxels)")
    plt.ylabel("Number of Detected Crystals")
    plt.show()
    np.save(f"{output_dir}/n_crystals.npy", n_crystals)
    np.save(f"{output_dir}/min_distances.npy", min_distances)
    
    MIN_MARKER_DISTANCE = int(input("Enter desired minimum marker distance based on the plot: "))
else:
    MIN_MARKER_DISTANCE = 15

local_maxima_coords = peak_local_max(
    distance_map,
    min_distance=MIN_MARKER_DISTANCE,
    labels=(data)  # Use the original boolean mask to exclude peaks in the background
)

# Create a marker image from the coordinates
markers_mask = np.zeros(distance_map.shape, dtype=bool)
markers_mask[tuple(local_maxima_coords.T)] = True
markers = label(markers_mask)[0]

print("Applying watershed algorithm...")
# The watershed algorithm "floods" from the markers to find boundaries
labels = watershed(-distance_map, markers, mask=data)

# --- 7. ANALYZE AND SAVE FINAL RESULTS ---

print(f"Segmentation complete. Found {labels.max()} unique crystals.")

np.save(f"{output_dir}/labeled_crystals.npy", labels)