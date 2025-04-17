#!/bin/bash

# Define dataset ID and configuration
DATASET_ID="001"
CONFIGURATION="3d_fullres"

# Loop through all 5 folds
for FOLD in {0..4}
do
    echo "Training model for fold ${FOLD}..."
    nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} --npz
done

echo "Training completed for all folds."
