import pandas as pd
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

class DataPreprocessing():
    def __init__(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame):
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset

    def get_training_dataset(self) -> pd.DataFrame:
        return self.training_dataset
    
    def get_testing_dataset(self) -> pd.DataFrame:
        return self.testing_dataset

    def encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        
        Encoding team and match results
        _param: df: the dataframe to be processed
        _return: df: the updated dataframe

        '''
        def team_encode(df):
            all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            team_encoded = one_hot_encoder.fit_transform(df[['HomeTeam', 'AwayTeam']])
            team_encoded_df = pd.DataFrame(team_encoded, columns=one_hot_encoder.get_feature_names_out(['HomeTeam', 'AwayTeam']))
            df = pd.concat([df.reset_index(drop=True), team_encoded_df.reset_index(drop=True)], axis=1)
            return df, team_encoded_df

        def match_encode(df):
            label_encoder = LabelEncoder()
            df["FTR_encoded"] = label_encoder.fit_transform(df["FTR"])
            return df
        
        df, team_encoded_df = team_encode(df)
        df = match_encode(df)

        return df
    


    def team_last_matches_performance(self, df: pd.DataFrame, team: str, 
                                      date: datetime, number_of_matches: int) -> tuple[float, int, float]:

        '''

        calculate the performance in recent matches
        _param: df : dataframe to be processed
                team: team to be analyzed
                date: the date of current match
                number_of_matches: number of matches of this team before current match
        _return:    avg_goal_diff: avg(goal_scored - goal_conceded)
                    points: points earned (3 for Win, 1 for Draw, 0 for Lose)
                    shot_on_target: avg shot on target 

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

    def add_team_performance(self, df: pd.DataFrame) -> pd.DataFrame:

        '''

        Add new columns about team performance
        _param: dataframe to be processed
        _return: updated dataframe

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
                                            +['HomeTeam_avg_goal_diff', 'HomeTeam_points', "HomeTeam_ShotOnTarget", 
                                            "AwayTeam_avg_goal_diff", "AwayTeam_points", "AwayTeam_ShotOnTarget"] 
                                            + dataset_columns[dataset_columns.index("FTHG"):]]

        return df

    def scale_betting_odd(self, df: pd.DataFrame) -> pd.DataFrame:

        '''

        Scale betting odd. Method: new_odd = (probability) / normalized_function
        _df: dataframe to be processed

        '''

        def normalize_betting_odd(df, columns):
            for col in columns:
                df[col] = df[col].apply(lambda x: 1/x)
            normalization_factor = df[columns].sum(axis=1)
            for col in columns:
                df[col] = df[col] / normalization_factor
            return df
        
        betting_comapnies = []
        for index in range(df.columns.get_loc("B365H"), df.columns.get_loc("MaxH"), 3):
            betting_comapnies.append(df.columns[index:index+3].tolist())

        for betting_odd in betting_comapnies:
            df = normalize_betting_odd(df, betting_odd)

        return df
    
    def scale(self, df, scaler: BaseEstimator) -> None:
        scaler = MinMaxScaler()
        columns_to_scale = df.loc[:, "HomeTeam_avg_goal_diff":"AwayTeam_ShotOnTarget"].columns
        df[columns_to_scale] = df[columns_to_scale].astype(float)
        df.loc[:, columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        return df

    def preprocessing(self, training_dataset: pd.DataFrame, 
                      testing_dataset: pd.DataFrame, 
                      scaler: BaseEstimator) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
        Perform data preprocessing on training dataset and testing dataset
        '''

        dataset = pd.concat([training_dataset, testing_dataset])

        dataset = self.encoding(dataset)

        dataset = self.add_team_performance(dataset)
        
        dataset = self.scale_betting_odd(dataset)
        dataset = dataset.drop(columns=["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "HF", "AF", "HY", "AY", "HR", "AR"])
        dataset = dataset.drop(columns=["HS", "AS", "HST", "AST", "HC", "AC", "Max>2.5", "Max<2.5", "AHh", "MaxAHH", "MaxAHA"])

        self.scale(dataset, scaler)

        pivot_date = testing_dataset["Date"][0]
        X_train = dataset.loc[dataset["Date"] < pivot_date, :].drop(columns = "FTR_encoded")
        X_test = dataset.loc[dataset["Date"] >= pivot_date, :].drop(columns = "FTR_encoded")

        y_train = dataset.loc[dataset["Date"] < pivot_date, ["FTR_encoded"]]
        y_test = dataset.loc[dataset["Date"] >= pivot_date, ["FTR_encoded"]]

        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        return X_train, X_test, y_train, y_test
