import pandas as pd


class DataCleaning():
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self.train = train.copy()
        self.test = test.copy()

    def get_training_ds(self) -> pd.DataFrame:
        return self.train
    
    def get_testing_df(self) -> pd.DataFrame:
        return self.test
    
    def parse_date(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" not in df.columns:
            return df
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
        if not df["Date"].is_monotonic_increasing:
            df = df.sort_values("Date", kind="mergesort")
        return df
    
    def nan_handling(self, df: pd.DataFrame, 
                     home_cols: list,
                     draw_cols: list,
                     away_cols: list,
                     avg_home: str = "BbAvH",
                     avg_draw: str = "BbAvD",
                     avg_away: str = "BbAvA"
    ) -> pd.DataFrame:
        
        out = df.copy()
        
        # Fill NaN values exactly as in the notebook
        for col in home_cols:
            if col in out.columns:
                out[col] = out[col].fillna(out[avg_home])
        
        for col in draw_cols:
            if col in out.columns:
                out[col] = out[col].fillna(out[avg_draw])
        
        for col in away_cols:
            if col in out.columns:
                out[col] = out[col].fillna(out[avg_away])

        return out
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = (
                out.columns.str.replace(r'^Bb', '', regex=True)
                .str.replace('Mx', 'Max')
                .str.replace('Av', 'Avg')
        )

        return out
    
    def column_alignment(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df_standardize = self.standardize_columns(train_df)
        test_df_standardize = self.standardize_columns(test_df)

        common = train_df_standardize.columns.intersection(test_df_standardize.columns)

        # Use the exact same approach as the notebook - just take the intersection
        return train_df_standardize[common].copy(), test_df_standardize[common].copy()
    
    def clean(self, home_cols: list, draw_cols: list, away_cols: list):
        self.train = self.parse_date(self.train)
        self.test = self.parse_date(self.test)

        self.train = self.nan_handling(self.train, home_cols, draw_cols, away_cols)

        self.train, self.test = self.column_alignment(self.train, self.test)
        return self