"""
Data Preprocessing Module for Hydrocephalus Medical Records.

This module handles data preprocessing including missing value imputation
using MICE (Multiple Imputation by Chained Equations) algorithm, feature
renaming, and data visualization.
"""

from typing import List

import matplotlib.pyplot as plt
import miceforest as mf
import numpy as np
import pandas as pd
import seaborn as sns

# Configure matplotlib for Chinese display
sns.set(font="Microsoft YaHei", font_scale=1.5)
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load medical data from Excel file.

    Args:
        filepath: Path to the Excel file

    Returns:
        Loaded DataFrame
    """
    return pd.read_excel(filepath, header=0, dtype=str)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Chinese columns to English for better compatibility.

    Args:
        df: Input DataFrame with Chinese column names

    Returns:
        DataFrame with renamed columns
    """
    column_mapping = {
        "血红蛋白": "Hemoglobin",
        "红细胞": "RBC_Count",
        "白蛋白": "Albumin",
        "白细胞": "CSF_WBC_Count",
        "葡萄糖": "CSF_Glucose",
        "蛋白": "CSF_Protein",
        "凝血酶原时间": "Prothrombin_Time",
        "国际标准化的比值": "INR",
        "活化部分凝血活酶时间": "APTT",
        "凝血酶时间": "Thrombin_Time",
        "纤维蛋白原": "Fibrinogen",
        "D-二聚体": "D_dimer",
    }

    df_renamed = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_renamed.columns:
            df_renamed.rename(columns={old_name: new_name}, inplace=True)

    return df_renamed


def preprocess_data_types(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert data types for specified columns.

    Args:
        df: Input DataFrame
        columns: List of column names to convert to numeric

    Returns:
        DataFrame with converted data types
    """
    df_processed = df.copy()

    # Convert specified columns to numeric
    for col in columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    # Convert infection column to integer if exists
    if "术后是否感染" in df_processed.columns:
        df_processed["术后是否感染"] = pd.to_numeric(
            df_processed["术后是否感染"], errors="coerce", downcast="integer"
        )

    return df_processed


def perform_mice_imputation(
    df: pd.DataFrame,
    num_datasets: int = 4,
    iterations: int = 3,
    random_state: int = 20,
) -> mf.ImputationKernel:
    """
    Perform MICE imputation on the dataset.

    Args:
        df: DataFrame with missing values
        num_datasets: Number of imputed datasets to generate
        iterations: Number of MICE iterations
        random_state: Random seed for reproducibility

    Returns:
        ImputationKernel object with imputed datasets
    """
    # Convert object columns to category type
    df_processed = df.copy()
    object_cols = df_processed.select_dtypes(include=["object"]).columns

    for col in object_cols:
        df_processed[col] = df_processed[col].astype("category")

    # Create imputation kernel
    kernel = mf.ImputationKernel(
        df_processed,
        datasets=num_datasets,
        save_all_iterations=True,
        save_models=1,
        random_state=random_state,
        mean_match_candidates=7,
    )

    # Perform MICE
    kernel.mice(iterations)

    return kernel


def visualize_imputation_results(kernel: mf.ImputationKernel, output_dir: str = "./data/Figure"):
    """
    Visualize MICE imputation results.

    Args:
        kernel: ImputationKernel with imputed data
        output_dir: Directory to save figures
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Plot imputed distributions
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    kernel.plot_imputed_distributions(wspace=0.4, hspace=0.6)
    plt.savefig(f"{output_dir}/imputed_distributions.jpg", dpi=500, bbox_inches="tight")
    plt.close()

    # Plot mean convergence
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    kernel.plot_mean_convergence(wspace=0.4, hspace=0.8)
    plt.savefig(f"{output_dir}/mean_convergence.jpg", dpi=500, bbox_inches="tight")
    plt.close()


def process_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process blood pressure column and categorize values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with processed blood pressure
    """
    df_processed = df.copy()

    for i in range(len(df_processed)):
        bp_value = df_processed.iloc[i, 6]
        if pd.notna(bp_value) and bp_value != "":
            try:
                blood = str(bp_value).split("/")
                if len(blood) == 2:
                    systolic = int(blood[0])
                    diastolic = int(blood[1])
                    if systolic > 140 or diastolic > 90:
                        df_processed.iloc[i, 7] = 2  # High BP
                    else:
                        df_processed.iloc[i, 7] = 1  # Normal BP
                else:
                    df_processed.iloc[i, 7] = 0  # Missing
            except (ValueError, IndexError):
                df_processed.iloc[i, 7] = 0  # Missing
        else:
            df_processed.iloc[i, 7] = 0  # Missing

    return df_processed


def main():
    """Main preprocessing pipeline."""
    # Load data
    print("Loading data...")
    df = load_data("./data/data-2022.6.13.xlsx")

    # Select relevant columns for imputation
    data_missing = df[df.columns[32:45]].copy()

    # Rename columns
    print("Renaming columns...")
    data_missing = rename_columns(data_missing)

    # Define columns for numeric conversion
    numeric_cols = [
        "Hemoglobin",
        "RBC_Count",
        "Albumin",
        "CSF_WBC_Count",
        "CSF_Glucose",
        "CSF_Protein",
        "Prothrombin_Time",
        "INR",
        "APTT",
        "Thrombin_Time",
        "Fibrinogen",
        "D_dimer",
    ]

    # Preprocess data types
    print("Converting data types...")
    data_missing = preprocess_data_types(data_missing, numeric_cols)

    # Perform MICE imputation
    print("Performing MICE imputation...")
    kernel = perform_mice_imputation(data_missing, num_datasets=4, iterations=3)

    # Visualize results
    print("Generating visualizations...")
    visualize_imputation_results(kernel)

    # Get completed datasets
    datasets = []
    for i in range(kernel.dataset_count()):
        datasets.append(kernel.complete_data(i))

    # Process blood pressure
    print("Processing blood pressure...")
    df = process_blood_pressure(df)

    # Save intermediate result
    df.to_excel("./data/2020.6.13.xlsx", index=False)

    # Combine imputed data from different datasets
    # Using different datasets for different features as per original implementation
    df.loc[:, "血红蛋白"] = datasets[0]["Hemoglobin"]
    df.loc[:, "红细胞"] = datasets[0]["RBC_Count"]
    df.loc[:, "白蛋白"] = datasets[3]["Albumin"]
    df.loc[:, "白细胞"] = datasets[0]["CSF_WBC_Count"]
    df.loc[:, "葡萄糖"] = datasets[2]["CSF_Glucose"]
    df.loc[:, "蛋白"] = datasets[2]["CSF_Protein"]
    df.loc[:, "凝血酶原时间"] = datasets[2]["Prothrombin_Time"]
    df.loc[:, "国际标准化的比值"] = datasets[2]["INR"]
    df.loc[:, "活化部分凝血活酶时间"] = datasets[2]["APTT"]
    df.loc[:, "凝血酶时间"] = datasets[1]["Thrombin_Time"]
    df.loc[:, "纤维蛋白原"] = datasets[2]["Fibrinogen"]
    df.loc[:, "D-二聚体"] = datasets[0]["D_dimer"]

    # Check for remaining missing values
    print("Missing values after imputation:")
    print(df[df.columns[32:45]].isnull().sum())

    # Save final processed data
    print("Saving processed data...")
    df.to_excel("./data/2020.7.11.xlsx", index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
