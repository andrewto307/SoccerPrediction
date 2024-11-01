import data_cleaning, data_collection, data_preprocessing, model


if __name__ == "__main__":
    data = data_collection.DataCollection('../data/training_file_paths.yaml', '../data/testing_file_paths.yaml')
    training_dataset, testing_dataset = data.data_collection(data.get_training_file_path(), data.get_testing_file_path())

    cleaning = data_cleaning.DataCleaning(training_dataset, testing_dataset)

    home_cols = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
    draw_cols = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
    away_cols = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]

    training_dataset, testing_dataset = cleaning.cleaning_data(training_dataset, testing_dataset,
                                        home_cols, draw_cols, away_cols)
    
    training_dataset.to_csv("../data/training_clean.csv")
    testing_dataset.to_csv('../data/test_clean.csv')