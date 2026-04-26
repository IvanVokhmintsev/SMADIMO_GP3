from typing import List

from langchain.tools import tool

def make_eda_tools(state):

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
    def drop_missing_target_rows(target_col: str) -> dict:
        """
        Удаляет строки, где отсутствует значение целевой переменной.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if target_col not in state.df.columns:
            return {
                "status": "error",
                "message": f"Target column '{target_col}' not found"
            }

        before_rows = len(state.df)

        state.df = state.df.dropna(subset=[target_col])

        after_rows = len(state.df)

        return {
            "status": "ok",
            "target_col": target_col,
            "dropped_rows": before_rows - after_rows,
            "remaining_rows": after_rows
        }

    @tool
    def drop_sparse_feature_rows(target_col: str, threshold: float = 0.95) -> dict:
        """
        Удаляет строки, где доля пропусков среди признаков больше или равна threshold.
        Целевая переменная не учитывается при подсчете пропусков.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if not 0 <= threshold <= 1:
            return {
                "status": "error",
                "message": "threshold must be between 0 and 1"
            }

        if target_col not in state.df.columns:
            return {
                "status": "error",
                "message": f"Target column '{target_col}' not found"
            }

        feature_cols = [col for col in state.df.columns if col != target_col]

        if not feature_cols:
            return {
                "status": "error",
                "message": "No feature columns found"
            }

        before_rows = len(state.df)

        missing_ratio = state.df[feature_cols].isnull().mean(axis=1)
        rows_to_keep = missing_ratio < threshold

        state.df = state.df.loc[rows_to_keep].copy()

        after_rows = len(state.df)

        return {
            "status": "ok",
            "target_col": target_col,
            "threshold": threshold,
            "dropped_rows": before_rows - after_rows,
            "remaining_rows": after_rows
        }

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


    return [drop_cols, analyze_missing, fill_missing,drop_missing_target_rows,
            drop_sparse_feature_rows, eda_summary]


