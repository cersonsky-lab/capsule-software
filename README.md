# Software for high-throughput analysis of capsules

This repository contains all the code used to generate the results in "Non-Destructive Pipeline for Analysis of Crystalline Solid Dispersions".

### 1. Installation
The environment can be installed using:
```
pip install -r requirements.txt
```

### 2. Capsule preprocessing

We provide two scripts to handle the processing steps discussed in our work. `src/preprocessing/remove_sample_holder.py` will, if used, solely remove the sample holder, and leave the inside of the capsule untouched. `src/preprocessing/preprocess.py` will do _everything_, it will remove the sample holder and label phases. Both of these output a large .npy file and accept a .nii file as input. They can be ran using:

```
python src/preprocessing/preprocess.py <nii_path>
```
where `nii_path` is the filepath to the .nii file.

### 3. Notebooks
Each of the associated Jupyter notebooks contains the code used to make each of the figures in the manuscript. These can be ran using the .npy files generated in (2) to obtain the results generated in the paper.
