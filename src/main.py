import data_cleaning, data_collection, data_preprocessing, model
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    data = data_collection.DataCollection('../data/training_file_paths.yaml', '../data/testing_file_paths.yaml')
    training_dataset, testing_dataset = data.data_collection(data.get_training_file_path(), data.get_testing_file_path())

    cleaning = data_cleaning.DataCleaning(training_dataset, testing_dataset)

    home_cols = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
    draw_cols = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
    away_cols = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]

    training_dataset, testing_dataset = cleaning.cleaning_data(cleaning.get_training_dataset(), cleaning.get_testing_dataset(),
                                        home_cols, draw_cols, away_cols)
    
    scaler_for_betting_odd = MinMaxScaler()
    preprocessing = data_preprocessing.DataPreprocessing(training_dataset, testing_dataset)
    X_train, X_test, y_train, y_test = preprocessing.preprocessing(preprocessing.get_training_dataset(), 
                                                                   preprocessing.get_testing_dataset(),
                                                                   scaler_for_betting_odd)
    
    X_train.to_csv("../data/train.csv")
    y_train.to_csv("../data/test.csv")