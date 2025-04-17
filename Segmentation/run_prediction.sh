#!/bin/bash

# Configuration
declare -A DATASETS=(
    ["001"]="Dataset001_Ascending"
    ["002"]="Dataset002_Sinuses"
    ["003"]="Dataset003_RootTop"
)

# Function to select dataset
select_dataset() {
    echo "Available datasets:"
    for id in "${!DATASETS[@]}"; do
        echo "$id: ${DATASETS[$id]}"
    done

    while true; do
        read -p "Enter dataset ID (001, 002, or 003): " DATASET_ID
        if [[ -n "${DATASETS[$DATASET_ID]}" ]]; then
            DATASET_NAME="${DATASETS[$DATASET_ID]}"
            break
        else
            echo "Invalid dataset ID. Please try again."
        fi
    done
}

# Main script
select_dataset

# Define paths based on selected dataset
INPUT_FOLDER="../nnUNet_raw/${DATASET_NAME}/imagesTs"
OUTPUT_BASE_FOLDER="../nnUNet_raw/Outputs/${DATASET_NAME}_predictions_fold"
ENSEMBLE_OUTPUT="../nnUNet_raw/Outputs/${DATASET_NAME}_Ensemble_Outputs"
CONFIGURATION="3d_fullres"

# Create output directory if it doesn't exist
mkdir -p "$ENSEMBLE_OUTPUT"

# Run inference for each fold
echo "Starting inference for ${DATASET_NAME}..."
for FOLD in {0..4}; do
    OUTPUT_FOLDER="${OUTPUT_BASE_FOLDER}${FOLD}"
    echo "Running inference for fold ${FOLD}..."
    nnUNetv2_predict -i "$INPUT_FOLDER" -o "$OUTPUT_FOLDER" -d "$DATASET_ID" -c "$CONFIGURATION" -f "$FOLD" --save_probabilities
done

# Run ensemble
echo "Running ensemble of all folds..."
FOLD_PATHS=()
for FOLD in {0..4}; do
    FOLD_PATHS+=("../nnUNet_raw/Outputs/${DATASET_NAME}_predictions_fold${FOLD}")
done

nnUNetv2_ensemble -i "${FOLD_PATHS[@]}" -o "$ENSEMBLE_OUTPUT" -np 4

# Verification
if [ -d "$ENSEMBLE_OUTPUT" ] && [ -n "$(ls -A "$ENSEMBLE_OUTPUT")" ]; then
    echo "Successfully completed ensemble for ${DATASET_NAME}"
    echo "Results saved to: $ENSEMBLE_OUTPUT"
else
    echo "Error: Ensemble output directory is empty or not created"
    exit 1
fi