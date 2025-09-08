import yaml
from pathlib import Path

import pandas as pd

class DataCollection:
    def __init__(self, training_file_path: str, testing_file_path: str):
        self.training_file_path = Path(training_file_path)
        self.testing_file_path = Path(testing_file_path)

    def get_training_file_path(self) -> str:
        return str(self.training_file_path)
    
    def get_testing_file_path(self) -> str:
        return str(self.testing_file_path)
        
    def load_file_paths(self, yaml_file_path: str) -> dict:
        with Path(yaml_file_path).open("r") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{yaml_file_path} : YAML must be a dict of  {{season: csv_path}}"
        return data

    def data_collection(self, training_file_paths: str, testing_file_paths: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        training_files = self.load_file_paths(training_file_paths)
        testing_files = self.load_file_paths(testing_file_paths)

        base_train_path = Path(training_file_paths).parent
        base_test_path = Path(testing_file_paths).parent

        training_dataframes = {}
        for key, value in training_files.items():
            raw = Path(value).expanduser()
            path = raw if raw.is_absolute() else (base_train_path / raw)
            if not path.exists():
                raise FileNotFoundError(f"Missing csv listed in {training_file_paths}: {path}")
            training_dataframes[key] = pd.read_csv(path)

        if not training_dataframes:
            raise ValueError(f"No csv listed in {training_file_paths}")

        dataframes_to_concat = [training_dataframes[key] for key in training_dataframes.keys()]
        training_dataset = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)

        testing_dataframes = {}
        for key, value in testing_files.items():
            raw = Path(value).expanduser()
            path = raw if raw.is_absolute() else (base_test_path / raw)
            if not path.exists():
                raise FileNotFoundError(f"Missing csv listed in {testing_file_paths}: {path}")
            testing_dataframes[key] = pd.read_csv(path)

        if not testing_dataframes:
            raise ValueError(f"No csv listed in {testing_file_paths}")
        
        dataframes_to_concat = [testing_dataframes[key] for key in testing_dataframes.keys()]
        testing_dataset = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)
        

        return training_dataset, testing_dataset