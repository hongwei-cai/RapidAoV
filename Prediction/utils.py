# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
# ---

"""
utils.py - Common helper functions for ascending aortic aneurysm research project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from typing import Dict, List, Tuple, Union
from functools import reduce
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc, r2_score, mean_squared_error, mean_absolute_error,
    classification_report
)


# --------------------------
# Data Loading & Preprocessing
# --------------------------

def load_data(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files into a dictionary of DataFrames
    
    Args:
        file_paths: Dictionary of {name: file_path} pairs
        
    Returns:
        Dictionary of {name: DataFrame} pairs
    """
    return {k: pd.read_csv(v) for k, v in file_paths.items()}

def clean_feature_df(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """
    Clean and rename feature columns in a radiomics DataFrame
    
    Args:
        df: Input DataFrame
        suffix: Suffix to add to column names
        
    Returns:
        Cleaned DataFrame
    """
    general_exclude_cols = ["Image", "Mask"] + [
        col for col in df.columns if "diagnostics" in col
    ]
    df = df.drop(
        columns=[col for col in df.columns if col in general_exclude_cols],
        errors='ignore'
    )
    return df.rename(
        columns={col: col + suffix for col in df.columns if col != "ID"}
    )

def merge_dataframes(df_dict: Dict[str, pd.DataFrame], 
                    selected_keys: List[str]) -> pd.DataFrame:
    """
    Merge multiple DataFrames on the 'ID' column
    
    Args:
        df_dict: Dictionary of DataFrames
        selected_keys: List of keys to merge
        
    Returns:
        Merged DataFrame
    """
    return reduce(
        lambda left, right: pd.merge(left, right, on="ID", how="left"),
        [df_dict[k] for k in selected_keys]
    )

# --------------------------
# Feature Engineering
# --------------------------

def categorize_diameter(value: float) -> int:
    """
    Categorize aortic diameter into clinical groups
    
    Args:
        value: Max diameter in mm
        
    Returns:
        0 (<40mm), 1 (40-45mm), 2 (45-50mm), or 3 (≥50mm)
    """
    if value < 40:
        return 0
    elif value < 45:
        return 1
    elif value < 50:
        return 2
    return 3

def filter_feature_types(df: pd.DataFrame, 
                        feature_type: str = "all") -> pd.DataFrame:
    """
    Filter features by type (shape+measure vs all)
    
    Args:
        df: Input DataFrame
        feature_type: "shape+measure" or "all"
        
    Returns:
        Filtered DataFrame
    """
    if feature_type == "shape+measure":
        texture_keywords = [
            "firstorder", "glcm", "glrlm", 
            "glszm", "gldm", "ngtdm"
        ]
        return df[[
            col for col in df.columns 
            if not any(key in col.lower() for key in texture_keywords) 
            or col in ["ID", "Label", "max_diameter"]
        ]]
    return df

# --------------------------
# Model Evaluation
# --------------------------

def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, np.ndarray, Dict[int, List]]:
    """
    Evaluate classifier performance
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        
    Returns:
        Tuple of (accuracy, confusion_matrix, roc_curves)
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    roc_curves = {}
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_curves[i] = (fpr, tpr, auc(fpr, tpr))
    
    return accuracy, cm, roc_curves

def evaluate_regressor(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, float, float]:
    """
    Evaluate regressor performance
    
    Args:
        model: Trained regressor
        X_test: Test features
        y_test: True values
        
    Returns:
        Tuple of (rmse, mae, r2)
    """
    y_pred = model.predict(X_test)
    return (
        np.sqrt(mean_squared_error(y_test, y_pred)),
        mean_absolute_error(y_test, y_pred),
        r2_score(y_test, y_pred)
    )

# --------------------------
# Visualization
# --------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: List[str] = None,
    save_path: str = None
) -> None:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        title: Plot title
        labels: List of class labels
        save_path: Path to save figure (optional)
    """
    if labels is None:
        labels = ["< 40 mm", "40 - 45 mm", "45 - 50 mm", "≥ 50 mm"]
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap="Blues", values_format=".0f")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches="tight")
    plt.show()

def plot_roc_curves(
    roc_data: Dict[int, List],
    title: str = "ROC Curves",
    save_path: str = None
) -> None:
    """
    Plot ROC curves for all classes
    
    Args:
        roc_data: Dictionary of ROC curve data
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    for class_id, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"Class {class_id} (AUROC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches="tight")
    plt.show()

def plot_feature_importance(
    model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
    importance_type: str = "weight",
    title: str = "Feature Importance",
    max_features: int = 10,
    save_path: str = None
) -> None:
    """
    Plot feature importance
    
    Args:
        model: Trained XGBoost model
        importance_type: Type of importance ("weight", "gain", "cover")
        title: Plot title
        max_features: Number of top features to show
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(8, 10))
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        max_num_features=max_features,
        title=title
    )
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches="tight")
    plt.show()

# --------------------------
# Model Configuration
# --------------------------

def get_xgb_classifier() -> xgb.XGBClassifier:
    """Get configured XGBoost classifier"""
    return xgb.XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=4,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.9,
        min_child_weight=2,
        gamma=0.1,
        random_state=42
    )

def get_xgb_regressor() -> xgb.XGBRegressor:
    """Get configured XGBoost regressor"""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        learning_rate=0.01,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.9,
        min_child_weight=2,
        gamma=0.1,
        random_state=42
    )
