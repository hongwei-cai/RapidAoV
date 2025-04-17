#!/bin/bash

# Define input and output directories
INPUT_FOLDER="../nnUNet_raw/Dataset001_Ascending/imagesTs"
OUTPUT_BASE_FOLDER="../nnUNet_raw/Outputs/Ascending_predictions_fold"
DATASET_ID="001"
CONFIGURATION="3d_fullres"

# Loop through all 5 folds
for FOLD in {0..4}
do
    OUTPUT_FOLDER="${OUTPUT_BASE_FOLDER}${FOLD}"
    echo "Running inference for fold ${FOLD}..."
    nnUNetv2_predict -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -d ${DATASET_ID} -c ${CONFIGURATION} -f ${FOLD} --save_probabilities
done

echo "Inference completed for all folds."
