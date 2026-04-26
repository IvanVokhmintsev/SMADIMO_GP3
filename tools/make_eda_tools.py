from typing import List

from langchain.tools import tool

def make_eda_tool(state):

    @tool
    def drop_cols(col_names: List[str]) -> dict:
        """
        Удаляет несколько колонок из DataFrame в state.
        """

        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        existing_cols = set(state.df.columns)

        to_drop = [col for col in col_names if col in existing_cols]
        missing = [col for col in col_names if col not in existing_cols]

        if not to_drop:
            return {
                "status": "error",
                "message": "No valid columns to drop"
            }

        state.df = state.df.drop(columns=to_drop)

        return {
            "status": "ok",
            "dropped": to_drop,
            "missing": missing,
            "remaining_columns": list(state.df.columns)
        }

    @tool
    def analyze_missing():
        """Analyze missing values in the dataset.

        Returns statistics about missing data per column, including count
        and percentage of missing values. This helps the agent decide how
        to handle incomplete data (drop, fill, or ignore).
        """
        df = state.df

        report = {
            "missing_count": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().mean() * 100).to_dict()
        }

        return report

    @tool
    def fill_missing(strategy: dict):
        """Fill or remove missing values in the dataset based on a strategy.

        Args:
            strategy: Dictionary mapping column names to imputation methods.
                      Supported methods:
                      - 'mean': fill with mean value
                      - 'median': fill with median value
                      - 'mode': fill with most frequent value
                      - 'drop': remove rows with missing values in that column
        """
        df = state.df.copy()

        for col, method in strategy.items():
            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif method == "drop":
                df = df.dropna(subset=[col])

        state.df = df
        return state

    @tool
    def eda_summary():
        """Generate a summary of the dataset for exploratory data analysis.

        Returns dataset shape, column types, missing values, duplicates,
        and descriptive statistics.
        """
        df = state.df

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_total": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "numeric_stats": df.describe().to_dict()
        }

    return [drop_cols, analyze_missing, fill_missing, eda_summary]


