import yaml

import pandas as pd

class DataCollection:
    def __init__(self, training_file_path: str, testing_file_path: str):
        self.training_file_path = training_file_path
        self.testing_file_path = testing_file_path

    def get_training_file_path(self) -> str:
        return self.training_file_path
    
    def get_testing_file_path(self) -> str:
        return self.testing_file_path
        
    def load_file_paths(self, yaml_file_path: str) -> dict:
        with open(yaml_file_path, 'r') as file:
            return yaml.safe_load(file)

    def data_collection(self, training_file_paths: str, testing_file_paths: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        training_files = self.load_file_paths(training_file_paths)
        testing_files = self.load_file_paths(testing_file_paths)
        
        training_dataframes = {}
        for key, value in training_files.items():
            training_dataframes[key] = pd.read_csv(value)

        dataframes_to_concat = [training_dataframes[key] for key in training_dataframes.keys()]
        training_dataset = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)

        testing_dataframes = {}
        for key, value in testing_files.items():
            testing_dataframes[key] = pd.read_csv(value)

        dataframes_to_concat = [testing_dataframes[key] for key in testing_dataframes.keys()]
        testing_dataset = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)
        

        return training_dataset, testing_dataset