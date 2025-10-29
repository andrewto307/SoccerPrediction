import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator


class DataPreprocessing:
    def __init__(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame):
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        
        # Elo rating parameters
        self.ELO_K = 20
        self.ELO_START = 1500
        self.ELO_HOME_ADV = 100

    def get_training_dataset(self) -> pd.DataFrame:
        return self.training_dataset
    
    def get_testing_dataset(self) -> pd.DataFrame:
        return self.testing_dataset

    def match_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Label encode match results (FTR column).
        
        Parameters:
            df: DataFrame containing FTR column
            
        Returns:
            DataFrame with FTR_encoded column added
        '''

        label_encoder = LabelEncoder()
        df["FTR_encoded"] = label_encoder.fit_transform(df["FTR"])
        return df

    def team_last_matches_performance(self, df: pd.DataFrame, team: str, 
                                    date: datetime, number_of_matches: int) -> tuple[float, int, float]:

        '''
        Calculate team performance in recent matches

        Parameters:
            df: DataFrame containing match data
            team: Team name to analyze
            date: Current match date
            number_of_matches: Number of recent matches to consider
            
        Returns:
            tuple: (avg_goal_diff, points, shot_on_target)
        '''

        past_n_matches = df.loc[((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) & 
                                (df["Date"] < date), :].tail(number_of_matches)

        goal_scored = (past_n_matches.loc[past_n_matches["HomeTeam"] == team, "FTHG"].sum() + 
                    past_n_matches.loc[past_n_matches["AwayTeam"] == team, "FTAG"].sum()
        )

        goals_conceded = (past_n_matches.loc[past_n_matches["HomeTeam"] == team, "FTAG"].sum() + 
                    past_n_matches.loc[past_n_matches["AwayTeam"] == team, "FTHG"].sum()
        )

        avg_goal_diff = (goal_scored - goals_conceded) / number_of_matches

        points = 0
        for _, match in past_n_matches.iterrows():
            if ((match["HomeTeam"] == team) and (match["FTR_encoded"] == 2)) or (
                (match["AwayTeam"] == team) and (match["FTR_encoded"] == 0)
            ):
                points += 3
            elif ((match["HomeTeam"] == team) and (match["FTR_encoded"] == 0)) or (
                (match["AwayTeam"] == team) and (match["FTR_encoded"] == 2)
            ):
                points += 0
            else:
                points += 1

        shot_on_target = (past_n_matches.loc[past_n_matches["HomeTeam"] == team, "HST"].sum() +
                        past_n_matches.loc[past_n_matches["AwayTeam"] == team, "AST"].sum()
        ) / number_of_matches
        
        return avg_goal_diff, points, shot_on_target

    def add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Add team performance features to the dataset

        Parameters:
            df: DataFrame containing match data
            
        Returns:
            DataFrame with team performance features added
        '''

        dataset_columns = df.columns.to_list()

        df[["HomeTeam_avg_goal_diff", "HomeTeam_points", "HomeTeam_ShotOnTarget"]] = df.apply(
            lambda row: pd.Series(
                self.team_last_matches_performance(df, row["HomeTeam"], row["Date"], 5)
            ),
            axis=1
        )

        df[["AwayTeam_avg_goal_diff", "AwayTeam_points", "AwayTeam_ShotOnTarget"]] = df.apply(
            lambda row: pd.Series(
                self.team_last_matches_performance(df, row["AwayTeam"], row["Date"], 5)
            ),
            axis=1
        )

        df = df[dataset_columns[:dataset_columns.index("FTHG")]
                + ['HomeTeam_avg_goal_diff', 'HomeTeam_points', "HomeTeam_ShotOnTarget", 
                   "AwayTeam_avg_goal_diff", "AwayTeam_points", "AwayTeam_ShotOnTarget"] 
                + dataset_columns[dataset_columns.index("FTHG"):]]

        return df

    # Elo rating

    def _outcome_scores(self, row: pd.Series) -> tuple[float, float]:
        '''
        Outcome scores of a match (1, 0.5, 0 for win, draw, loss)

        Parameters:
            row: Series containing match data
            
        Returns:
            tuple: (home_score, away_score)
        '''

        if "FTR" in row and pd.notna(row["FTR"]):
            if row["FTR"] == "H":   return 1.0, 0.0
            if row["FTR"] == "A":   return 0.0, 1.0
            return 0.5, 0.5

        enc = row.get("FTR_encoded", None)
        if enc == 2:    return 1.0, 0.0   # Home win
        if enc == 0:    return 0.0, 1.0   # Away win
        if enc == 1:    return 0.5, 0.5   # Draw

        return 0.5, 0.5

    def _expected_scores(self, R_home: float, R_away: float, home_adv: float = None) -> tuple[float, float]:
        '''
        Expected scores of 2 teams in a match

        Parameters:
            R_home: Elo rating of home team
            R_away: Elo rating of away team
            home_adv: Home advantage
            
        Returns:
            tuple: (expected_home_score, expected_away_score)
        '''

        if home_adv is None:
            home_adv = self.ELO_HOME_ADV
            
        E_home = 1.0 / (1.0 + 10 ** ((R_away - (R_home + home_adv)) / 400.0))
        return E_home, 1.0 - E_home

    def add_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Elo ratings for home and away teams.
        
        Args:
            df: DataFrame containing match data
            
        Returns:
            DataFrame with home_elo and away_elo columns added
        """
        if not {"HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
            raise ValueError("Dataset must contain 'HomeTeam', 'AwayTeam', and 'Date' columns.")

        df_sorted = df.sort_values("Date", kind="mergesort").copy()
        ratings = {}  

        home_elos = []
        away_elos = []

        for _, row in df_sorted.iterrows():
            h = row["HomeTeam"]
            a = row["AwayTeam"]

            Rh = ratings.get(h, self.ELO_START)
            Ra = ratings.get(a, self.ELO_START)

            # record pre-match
            home_elos.append(Rh)
            away_elos.append(Ra)

            Eh, Ea = self._expected_scores(Rh, Ra, home_adv=self.ELO_HOME_ADV)
            Sh, Sa = self._outcome_scores(row)

            # Elo updates
            ratings[h] = Rh + self.ELO_K * (Sh - Eh)
            ratings[a] = Ra + self.ELO_K * (Sa - Ea)

        df_sorted["home_elo"] = home_elos
        df_sorted["away_elo"] = away_elos

        df_with_elo = df_sorted.sort_index()
        return df_with_elo

    def normalize_betting_odds(self, df: pd.DataFrame, columns: list, 
                              prefix: str = None, keep_overround: bool = True) -> pd.DataFrame:
        '''
        Normalize betting odds
        
        Parameters:
            df: DataFrame containing betting odds
            columns: List of columns to normalize
            prefix: Prefix of the betting company
            keep_overround: Whether to keep the overround for analysis
        '''

        if not all(c in df.columns for c in columns):
            return df

        # avoid 1/0
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].mask(df[col] <= 0)

        valid = df[columns].notna().all(axis=1)
        if not valid.any():
            return df

        # Implied probs
        inv = 1.0 / df.loc[valid, columns].astype(float)
        if keep_overround and prefix is not None:
            df.loc[valid, f"{prefix}_overround"] = inv.sum(axis=1)  # S (sum of implied probs)

        # Final renormalization so H+D+A = 1
        denom = inv.sum(axis=1)
        for col in columns:
            df.loc[valid, col] = inv[col] / denom

        return df

    def normalize_all_betting_odds(self, df: pd.DataFrame) -> pd.DataFrame:

        '''
        Add normalized betting odds to the dataset
        
        Parameters:
            df: DataFrame containing betting odds
            
        Returns:
            DataFrame with normalized betting odds added
        '''

        betting_companies = ["B365", "BW", "IW", "WH", "VC", "Max", "PS", "PSC"]
        for p in betting_companies:
            triplet = [f"{p}{s}" for s in ("H", "D", "A")]
            df = self.normalize_betting_odds(df, triplet, prefix=p, keep_overround=True)
        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features
        Creates: pH_mean, pD_mean, pA_mean, home_adv, draw_tightness, elo_diff
        """
        out = df.copy()

        candidates = ["B365", "VC", "IW", "Max", "WH", "BW", "PS", "PSC"]
        bookies = [p for p in candidates if all(f"{p}{s}" in out.columns for s in ("H", "D", "A"))]

        if not bookies:
            raise ValueError("No bookmaker triplets (H/D/A) found to build consensus features.")

        H_cols = [f"{p}H" for p in bookies]
        D_cols = [f"{p}D" for p in bookies]
        A_cols = [f"{p}A" for p in bookies]

        # 2) Consensus means (market consensus)
        out["pH_mean"] = out[H_cols].mean(axis=1)
        out["pD_mean"] = out[D_cols].mean(axis=1)
        out["pA_mean"] = out[A_cols].mean(axis=1)

        # 3) Engineered odds features
        out["home_adv"] = out["pH_mean"] - out["pA_mean"]
        out["draw_tightness"] = 1.0 - (out["pH_mean"] + out["pA_mean"])

        # 4) Elo feature
        if "home_elo" in out.columns and "away_elo" in out.columns:
            out["elo_diff"] = out["home_elo"] - out["away_elo"]
        else:
            out["elo_diff"] = 0.0

        return out

    def scale_features(self, df: pd.DataFrame, scaler: BaseEstimator) -> pd.DataFrame:
        """
        Scale team performance features using the provided scaler.
        
        Parameters:
            df: DataFrame to scale
            scaler: Scaler instance to use
            
        Returns:
            DataFrame with scaled features
        """
        columns_to_scale = df.loc[:, "HomeTeam_avg_goal_diff":"AwayTeam_ShotOnTarget"].columns
        df[columns_to_scale] = df[columns_to_scale].astype(float)
        df.loc[:, columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary columns from the dataset.
        
        Parameters:
            df: DataFrame to clean
            
        Returns:
            DataFrame with unnecessary columns removed
        """
        df = df.drop(columns=["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", 
                              "HF", "AF", "HY", "AY", "HR", "AR"])
        
        df = df.drop(columns=["HS", "AS", "HST", "AST", "HC", "AC", 
                              "Max>2.5", "Max<2.5", "AHh", "MaxAHH", "MaxAHA"])
        
        return df

    def preprocessing(self, training_dataset, testing_dataset, scaler):
        # 1) concat then deterministic encoding/features (same as before)
        dataset = pd.concat([training_dataset, testing_dataset])
        dataset = self.match_encode(dataset)
        dataset = self.add_team_performance_features(dataset)
        dataset = self.add_elo_ratings(dataset)
        dataset = self.normalize_all_betting_odds(dataset)

        # 2) (optional) drop obvious target-only columns BEFORE split (same as you do)
        dataset = dataset.drop(columns=["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
                                        "HF", "AF", "HY", "AY", "HR", "AR"])

        # 3) Choose a stable split boundary
        pivot_date = testing_dataset["Date"].min()  # instead of .iloc[0]
        X = dataset.drop(columns="FTR_encoded")
        y = dataset["FTR_encoded"]

        X_train = X.loc[X["Date"] < pivot_date].copy()
        X_test  = X.loc[X["Date"] >= pivot_date].copy()
        y_train = y.loc[X["Date"] < pivot_date].copy()
        y_test  = y.loc[X["Date"] >= pivot_date].copy()

        # 4) Scale ONLY the intended columns, fit on train, transform test
        cols_to_scale = X_train.loc[:, "HomeTeam_avg_goal_diff":"AwayTeam_ShotOnTarget"].columns
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale].astype(float))
        X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale].astype(float))

        # 5) Add engineered features AFTER scaling (same as before)
        X_train = self.add_engineered_features(X_train)
        X_test  = self.add_engineered_features(X_test)

        # 6) Drop the remaining columns (same list you had after scaling)
        drop_cols = ["HS", "AS", "HST", "AST", "HC", "AC", "Max>2.5", "Max<2.5", "AHh", "MaxAHH", "MaxAHA"]
        X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
        X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

        # 7) Persist column order for reproducibility
        feature_columns = X_train.columns
        feature_columns.to_series(name="feature").to_csv("../data/feature_columns.csv", index=False)
        print(f"Saved feature_columns.csv with {len(feature_columns)} columns")

        return X_train, X_test, y_train, y_test
