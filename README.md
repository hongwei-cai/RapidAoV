# RapidAoV
Rapid Diagnosis of Aortic Valve Insuﬃciency in Aortic Root Enlargement

# nnU-Net Segmentation Pipeline

This repository contains a complete pipeline for preparing data for nnU-Net, including resampling images and labels, remapping label values, creating the `dataset.json` file, and visualizing overlays.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
  - [1. Downsampling](#1-downsampling)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Label Remapping](#3-label-remapping)
  - [4. Direction Consistency Check](#4-direction-consistency-check)
  - [5. Dataset JSON Creation](#5-dataset-json-creation)
  - [6. Visualization](#6-visualization)
- [Next Steps](#next-steps)

---

## Introduction

This pipeline prepares medical imaging datasets for nnU-Net, a state-of-the-art segmentation framework. It includes:
- Fetching and splitting the original data.
- Downsampling images and labels to a target spacing.
- Remapping label values for consistency.
- Creating the `dataset.json` file required by nnU-Net.
- Visualizing overlays of images and labels.

---

## Setup

### Prerequisites
- Python 3.10 or higher
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `SimpleITK`

Install the required libraries using:
```bash
pip install numpy matplotlib SimpleITK
```

---

## Pipeline Overview

### 1. Downsampling

Images and labels are resampled to a target spacing of `(2.0, 2.0, 2.0)` using the `SimpleITK` library.

Functions:
- `downsample_images()`: Resamples all images in the `../../Data/images` directory.
- `downsample_labels(dataset_type)`: Resamples labels for a specific dataset type (`Ascending`, `Sinuses`, or `RootTop`).

Output:
- Resampled images are saved in `../Resampled/Common_2mm_images`.
- Resampled labels are saved in `../Resampled/{dataset_type}_2mm_labels`.

---

### 2. Data Preparation

The `prepare_data(dataset_type)` function:
- Splits the resampled data into training and testing sets.
- Copies the split data into nnU-Net-compatible directories:
  - `../nnUNet_raw/Dataset{dataset_number}_{dataset_type}/imagesTr`
  - `../nnUNet_raw/Dataset{dataset_number}_{dataset_type}/imagesTs`
  - `../nnUNet_raw/Dataset{dataset_number}_{dataset_type}/labelsTr`
  - `../nnUNet_raw/Dataset{dataset_number}_{dataset_type}/labelsTs`

---

### 3. Label Remapping

The `process_labels(dataset_type)` function remaps label values for each dataset type:
- `Ascending`: Maps label `4` to `1`.
- `Sinuses`: Keeps labels `1`, `2`, and `3` unchanged.
- `RootTop`: Maps label `5` to `1`.

---

### 4. Direction Consistency Check

The `check_origins(images_dir, labels_dir, image_files, label_files)` function ensures that the origin, spacing, and direction of images and labels are consistent.

---

### 5. Dataset JSON Creation

The `create_dataset_json(dataset_type)` function generates the `dataset.json` file required by nnU-Net. It includes:
- Channel names.
- Label mappings.
- Training and testing file paths.

---

### 6. Visualization

The `plot_random_overlay(dataset_type)` function visualizes overlays of images and labels in axial, coronal, and sagittal planes.

---

## Next Steps

1. **Plan and Preprocess the Dataset**:
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)
   ```
   Example:
   ```bash
   nnUNetv2_plan_and_preprocess -d 001
   ```

2. **Train the Model**:
   ```bash
   nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD
   ```
   Example:
   ```bash
   nnUNetv2_train 001 3d_fullres 0 --npz && \
   nnUNetv2_train 001 3d_fullres 1 --npz && \
   nnUNetv2_train 001 3d_fullres 2 --npz && \
   nnUNetv2_train 001 3d_fullres 3 --npz && \
   nnUNetv2_train 001 3d_fullres 4 --npz
   ```

3. **Run Predictions**:
   ```bash
   nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET -c CONFIGURATION -f FOLD --save_probabilities
   ```
   Example:
   ```bash
   nnUNetv2_predict -i nnUNet_raw/Dataset001_Ascending/imagesTr -o nnUNet_outputs/Ascending_predictions -d 001 -c 3d_fullres -f 0 --save_probabilities
   ```

---

## Summary

This pipeline prepares datasets for nnU-Net segmentation tasks. It ensures data consistency, generates required files, and provides visualization tools for quality assurance.
