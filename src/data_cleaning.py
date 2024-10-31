import pandas as pd
import numpy as np

import yaml

def load_file_paths(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

def data_collection(training_files, test_file):

    training_dataframes = {}
    for key, value in training_files.items():
        training_dataframes[key] = pd.read_csv(value)

    dataframes_to_concat = [training_dataframes[key] for key in training_dataframes.keys()]
    training_dataset = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)

    testing_dataset = pd.read_csv("../data/19-20.csv")

    return training_dataset, testing_dataset

def nan_handling(df: pd.DataFrame, home_cols: list, draw_cols: list, away_cols: list):
    """Fill NaN values in specific columns using averages."""
    try:
        for col in home_cols:
            df[col] = df[col].fillna(df["BbAvH"])

        for col in draw_cols:
            df[col] = df[col].fillna(df["BbAvD"])

        for col in away_cols:
            df[col] = df[col].fillna(df["BbAvA"])
    except KeyError as e:
        print(f"Column missing for NaN handling: {e}")

    return df

def column_alignment(training_dataset, testing_dataset):
    def standardize_columns(df):
        return (
            df.columns.str.replace(r'^Bb', '', regex=True)
              .str.replace('Mx', 'Max')
              .str.replace('Av', 'Avg')
        )

    # Standardize column names in both datasets
    training_dataset.columns = standardize_columns(training_dataset)
    testing_dataset.columns = standardize_columns(testing_dataset)

    # Align datasets by common columns
    common_columns = training_dataset.columns.intersection(testing_dataset.columns)
    training_dataset = training_dataset[common_columns]
    testing_dataset = testing_dataset[common_columns]

    return training_dataset, testing_dataset

def cleaning_data(training_files, test_file):

    file_paths = load_file_paths(training_files)

    training_dataset, testing_dataset = data_collection(file_paths, test_file)

    home_cols = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
    draw_cols = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
    away_cols = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]

    training_dataset = nan_handling(training_dataset, home_cols, draw_cols, away_cols)
    training_dataset, testing_dataset = column_alignment(training_dataset, testing_dataset)

    return training_dataset, testing_dataset

training_dataset, testing_dataset = cleaning_data("../data/file_paths.yaml", "../data/19-20.csv")

training_dataset.to_csv("../data/training_clean.csv")
testing_dataset.to_csv("../data/testing_clean.csv")