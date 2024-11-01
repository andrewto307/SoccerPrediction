import pandas as pd
import numpy as np

import yaml


class DataCleaning:
    def __init__(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame) -> None:
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset


    def nan_handling(self, df: pd.DataFrame, home_cols: list, draw_cols: list, away_cols: list) -> pd.DataFrame:
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

    def column_alignment(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def cleaning_data(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame, 
                      home_cols: list, draw_cols: list, away_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:

        training_dataset = self.nan_handling(training_dataset, home_cols, draw_cols, away_cols)
        training_dataset, testing_dataset = self.column_alignment(training_dataset, testing_dataset)

        return training_dataset, testing_dataset