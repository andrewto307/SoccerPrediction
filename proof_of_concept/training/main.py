"""
Main entry point for the Soccer Prediction System.
This script handles the complete pipeline: data collection, cleaning, preprocessing, and model training.
"""

import sys
import logging
from pathlib import Path

# This module lives in proof_of_concept/training/. Put its own directory on the path
# (for the sibling trainer modules) and the repo's src/ (for the shared data_* and
# model_configs modules that the live pipeline also uses).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import data_collection
import data_cleaning
import data_preprocessing
import model
from sklearn.preprocessing import MinMaxScaler
import argparse

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # shared repo data/ (see path shim above)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """
    Data pipeline: data collection -> cleaning -> preprocessing
    """
    logger.info("Soccer Prediction System - Data Pipeline")

    try:
        # 1) Data Collection
        logger.info("Collecting data...")
        data = data_collection.DataCollection(
            str(DATA_DIR / "training.yaml"),
            str(DATA_DIR / "testing.yaml"),
        )
        training_dataset, testing_dataset = data.data_collection(data.get_training_file_path(), data.get_testing_file_path())
        logger.info("Collected: %d training samples, %d test samples", training_dataset.shape[0], testing_dataset.shape[0])

        # 2) Data Cleaning
        logger.info("Cleaning data...")
        cleaning = data_cleaning.DataCleaning(training_dataset, testing_dataset)

        # Use column configuration
        home_cols = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
        draw_cols = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
        away_cols = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]

        cleaning.clean(home_cols, draw_cols, away_cols)
        training_dataset = cleaning.get_training_ds()
        testing_dataset = cleaning.get_testing_df()

        # 3) Data Preprocessing
        logger.info("Preprocessing data...")
        scaler_for_betting_odd = MinMaxScaler()
        preprocessing = data_preprocessing.DataPreprocessing(training_dataset, testing_dataset)
        X_train, X_test, y_train, y_test = preprocessing.preprocessing(
            preprocessing.get_training_dataset(),
            preprocessing.get_testing_dataset(),
            scaler_for_betting_odd
        )

        # 4) Save processed datasets
        logger.info("Saving processed datasets...")
        X_train.to_csv(DATA_DIR / "X_train_test.csv")
        y_train.to_csv(DATA_DIR / "y_train_test.csv")
        X_test.to_csv(DATA_DIR / "X_test_test.csv")
        y_test.to_csv(DATA_DIR / "y_test_test.csv")

        logger.info("Data pipeline completed successfully!")

        data_dir = str(DATA_DIR)

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
            logger.info("Training model...")
            spm = model.SoccerPredictionModel(model_type=args.model_type)

            X_tr, X_te = X_train.copy(), X_test.copy()
            y_tr, y_te = y_train.squeeze().copy(), y_test.squeeze().copy()

            feature_set_order = [args.feature_set, "odds_form_teams_elo", "odds_form_teams"]
            last_err = None
            for fs in feature_set_order:
                try:
                    logger.info("Using feature set: %s", fs)
                    spm.train(X_tr, y_tr, X_te, y_te,
                              feature_set=fs,
                              apply_smote=not args.no_smote)
                    break
                except KeyError as e:
                    last_err = e
                    logger.warning("Missing columns for feature set '%s': %s. Trying fallback...", fs, e)
                    continue
            else:
                raise last_err

            metrics = spm.evaluate(X_te, y_te)
            logger.info("Metrics: %s", metrics)
            if args.save_model:
                save_path = Path(args.save_model)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                spm.save_model(str(save_path))
                logger.info("Saved trained model to %s", save_path)
        else:
            logger.info("Skipping model training (use --train-model to enable).")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
