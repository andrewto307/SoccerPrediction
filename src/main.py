"""
Main entry point for the Soccer Prediction System.
This script handles the complete pipeline: data collection, cleaning, preprocessing, and model training.
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import data_collection
import data_cleaning
import data_preprocessing
import model
from sklearn.preprocessing import MinMaxScaler
import argparse


def main():
    """
    Data pipeline: data collection -> cleaning -> preprocessing
    """
    print("⚽ Soccer Prediction System - Data Pipeline")
    print("=" * 50)
    
    try:
        # 1) Data Collection
        print("📊 Collecting data...")
        data = data_collection.DataCollection('../data/training.yaml', '../data/testing.yaml')
        training_dataset, testing_dataset = data.data_collection(data.get_training_file_path(), data.get_testing_file_path())
        print(f"   Collected: {training_dataset.shape[0]} training samples, {testing_dataset.shape[0]} test samples")
        
        # 2) Data Cleaning
        print("🧹 Cleaning data...")
        cleaning = data_cleaning.DataCleaning(training_dataset, testing_dataset)
        
        # Use column configuration 
        home_cols = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
        draw_cols = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
        away_cols = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]
        
        # Clean the data
        cleaning.clean(home_cols, draw_cols, away_cols)
        training_dataset = cleaning.get_training_ds()
        testing_dataset = cleaning.get_testing_df()
        
        # 3) Data Preprocessing
        print("⚙️ Preprocessing data...")
        scaler_for_betting_odd = MinMaxScaler()
        preprocessing = data_preprocessing.DataPreprocessing(training_dataset, testing_dataset)
        X_train, X_test, y_train, y_test = preprocessing.preprocessing(
            preprocessing.get_training_dataset(), 
            preprocessing.get_testing_dataset(),
            scaler_for_betting_odd
        )
        
        # 4) Save processed datasets
        print("💾 Saving processed datasets...")
        X_train.to_csv("../data/X_train_test.csv")
        y_train.to_csv("../data/y_train_test.csv")
        X_test.to_csv("../data/X_test_test.csv")
        y_test.to_csv("../data/y_test_test.csv")
        
        print("Data pipeline completed successfully!")

        data_dir = "../data/"

        parser = argparse.ArgumentParser(add_help=False)
        # These flags only matter if you run: python main.py --train-model
        parser.add_argument("--train-model", action="store_true",
                            help="Train a model immediately after preprocessing")
        parser.add_argument("--model-type", default="catboost",
                            help="Model type (catboost, random_forest, gradient_boosting, naive_bayes, stacking)")
        parser.add_argument("--feature-set", default="odds_form_teams",
                            help="Feature set (odds_form_teams | odds_form_teams_elo | odds_form_teams_elo_consensus)")
        parser.add_argument("--no-smote", action="store_true",
                            help="Disable SMOTENC balancing if using non-CatBoost models")
        parser.add_argument("--save-model", default=None,
                            help="Path to save the trained model (e.g., ../models/model.pkl)")
        parser.add_argument("--data-dir", default=str(data_dir),
                            help="Directory where X_train.csv, etc. were saved")
        args, _unknown = parser.parse_known_args()

        if args.train_model:
            print("Training model...")
            spm = model.SoccerPredictionModel(model_type=args.model_type)
            
            # Use the freshly preprocessed in-memory datasets to avoid loading stale CSVs
            X_tr, X_te = X_train.copy(), X_test.copy()
            y_tr, y_te = y_train.squeeze().copy(), y_test.squeeze().copy()

            # Try requested feature set first, then fall back progressively if some columns are missing
            feature_set_order = [args.feature_set, "odds_form_teams_elo", "odds_form_teams"]
            last_err = None
            for fs in feature_set_order:
                try:
                    print(f"   ➤ Using feature set: {fs}")
                    spm.train(X_tr, y_tr, X_te, y_te,
                              feature_set=fs,
                              apply_smote=not args.no_smote)
                    break
                except KeyError as e:
                    # Missing columns for this feature set, try a leaner one
                    last_err = e
                    print(f"   ⚠️ Missing columns for feature set '{fs}': {e}. Trying fallback...")
                    continue
            else:
                # Exhausted all options
                raise last_err

            metrics = spm.evaluate(X_te, y_te)
            print("📊 Metrics:", metrics)
            if args.save_model:
                save_path = Path(args.save_model)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                spm.save_model(str(save_path))
                print(f"💾 Saved trained model to {save_path}")
        else:
            print("Skipping model training (use --train-model to enable).")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
