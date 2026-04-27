from typing import List, Dict

import pandas as pd
from langchain.tools import tool




def make_eda_tools(state):
    @tool
    def columns_description() -> dict:
        """Return structured metadata about all dataset columns.

        Provides column-level context so the agent can understand the role
        of each feature and make decisions such as dropping identifiers,
        selecting categorical columns, handling missing values, or creating
        new features.

        Returns:
            Dictionary where each key is a column name and each value contains:
            - dtype: pandas data type
            - inferred_role: guessed semantic role
              (id, numeric, categorical, boolean, datetime, text)
            - missing_count: number of missing values
            - missing_percent: percent of missing values
            - unique_count: number of unique values
            - unique_ratio: unique_count / rows
            - top_values: most frequent values with counts
            - sample_values: example values
            - min / max / mean / median (for numeric columns)
        """
        df = state.df

        if df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        result = {}
        n_rows = len(df)

        for col in df.columns:
            s = df[col]

            dtype = str(s.dtype)
            missing_count = int(s.isna().sum())
            missing_percent = round((missing_count / max(n_rows, 1)) * 100, 2)
            unique_count = int(s.nunique(dropna=True))
            unique_ratio = round(unique_count / max(n_rows, 1), 4)

            top_values = (
                s.value_counts(dropna=True)
                .head(5)
                .to_dict()
            )

            sample_values = (
                s.dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

            # -------- role inference --------
            role = "unknown"

            lower_name = col.lower()

            if "id" == lower_name or lower_name.endswith("_id") or lower_name.startswith("id_"):
                role = "id"

            elif pd.api.types.is_bool_dtype(s):
                role = "boolean"

            elif pd.api.types.is_datetime64_any_dtype(s):
                role = "datetime"

            elif pd.api.types.is_numeric_dtype(s):
                if unique_count <= 2:
                    role = "boolean"
                else:
                    role = "numeric"

            elif dtype == "object":
                avg_len = s.dropna().astype(str).str.len().mean() if not s.dropna().empty else 0

                if unique_ratio > 0.9:
                    role = "id"

                elif avg_len > 25:
                    role = "text"

                else:
                    role = "categorical"

            else:
                role = "categorical"

            info = {
                "dtype": dtype,
                "inferred_role": role,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "top_values": top_values,
                "sample_values": sample_values,
            }

            # -------- numeric stats --------
            if pd.api.types.is_numeric_dtype(s):
                info["min"] = float(s.min()) if not s.dropna().empty else None
                info["max"] = float(s.max()) if not s.dropna().empty else None
                info["mean"] = float(s.mean()) if not s.dropna().empty else None
                info["median"] = float(s.median()) if not s.dropna().empty else None

            result[col] = info

        return {
            "status": "ok",
            "rows": n_rows,
            "columns": len(df.columns),
            "column_info": result
        }

    @tool
    def drop_cols(col_names: List[str]) -> dict:
        """
        Drop multiple columns from the DataFrame stored in state.
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
    def analyze_missing() -> dict:
        """Analyze missing values in the dataset.

        Returns statistics about missing data per column, including count
        and percentage of missing values. This helps the agent decide how
        to handle incomplete data (drop, fill, or ignore).
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df

        return  {
            "status": "ok",
            "missing_count": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().mean() * 100).to_dict()
        }

    @tool
    def fill_missing(strategy: Dict[str, str]) -> dict:
        """Fill or remove missing values in the dataset based on a strategy.

        Args:
            strategy: Dictionary mapping column names to imputation methods.
                      Supported methods:
                      - 'mean': fill with mean value
                      - 'median': fill with median value
                      - 'mode': fill with most frequent value
                      - 'drop': remove rows with missing values in that column
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if strategy is None:
            return {
                "status": "error",
                "message": "Strategy is empty"
            }
        df = state.df.copy()
        actions = {}

        for col, method in strategy.items():
            if col not in df.columns:
                actions[col] = "column_not_found"
                continue

            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                mode = df[col].mode()
                if mode.empty:
                    actions[col] = "mode_not_available"
                    continue
                df[col] = df[col].fillna(mode[0])
            elif method == "drop":
                df = df.dropna(subset=[col])
            else:
                actions[col] = "unsupported_method"
                continue
            actions[col] = method

        state.df = df

        return {
            "status": "ok",
            "actions": actions,
            "shape_after": df.shape
        }

    @tool
    def eda_summary()-> dict:
        """Generate a summary of the dataset for exploratory data analysis.

        Returns dataset shape, column types, missing values, duplicates,
        and descriptive statistics.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df

        return {
            "status": "ok",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_total": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "numeric_stats": df.describe().to_dict()
        }

    @tool
    def analyze_duplicates() -> dict:
        """Analyze duplicate rows in the dataset.

        Returns the number and percentage of duplicated rows.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df
        duplicate_count = int(df.duplicated().sum())

        return {
            "status": "ok",
            "duplicate_count": duplicate_count,
            "duplicate_percent": round(duplicate_count / len(df) * 100, 2)
            if len(df) > 0 else 0
        }

    @tool
    def drop_duplicates() -> dict:
        """Remove duplicate rows from the dataset."""
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        rows_before = len(state.df)
        state.df = state.df.drop_duplicates()
        rows_after = len(state.df)

        return {
            "status": "ok",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "removed_duplicates": rows_before - rows_after
        }

    @tool
    def analyze_outliers() -> dict:
        """Detect outliers in numeric columns using the IQR method.

        Returns outlier count, percentage, and IQR bounds for each numeric column.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df
        numeric_cols = df.select_dtypes(include="number").columns
        report = {}

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            report[col] = {
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / len(df) * 100, 2)
                if len(df) > 0 else 0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }

        return {
            "status": "ok",
            "method": "IQR",
            "outliers": report
        }

    @tool
    def handle_outliers(strategy: Dict[str, str]) -> dict:
        """Handle outliers in numeric columns using a column-level strategy.

        Args:
            strategy: Dictionary mapping column names to methods.
                      Supported methods:
                      - 'drop': remove rows with outliers
                      - 'cap': cap outliers to IQR lower and upper bounds
                      - 'ignore': leave outliers unchanged
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if strategy is None:
            return {
                "status": "error",
                "message": "Strategy is empty"
            }

        df = state.df.copy()
        actions = {}

        for col, method in strategy.items():
            if col not in df.columns:
                actions[col] = "column_not_found"
                continue

            if col not in df.select_dtypes(include="number").columns:
                actions[col] = "not_numeric"
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if method == "drop":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == "cap":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "ignore":
                pass
            else:
                actions[col] = "unsupported_method"
                continue

            actions[col] = method

        state.df = df

        return {
            "status": "ok",
            "actions": actions,
            "shape_after": df.shape
        }

    @tool
    def analyze_target_balance(target_col: str) -> dict:
        """Analyze target variable distribution.

        Useful for classification tasks to detect class imbalance.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df

        if target_col not in df.columns:
            return {
                "status": "error",
                "message": f"Target column '{target_col}' not found"
            }

        counts = df[target_col].value_counts(dropna=False)
        percents = df[target_col].value_counts(normalize=True, dropna=False) * 100

        return {
            "status": "ok",
            "target_column": target_col,
            "class_counts": counts.to_dict(),
            "class_percent": percents.round(2).to_dict(),
            "is_imbalanced": bool(percents.max() > 70)
        }

    @tool
    def apply_preprocessing_plan(plan: Dict) -> dict:
        """Apply a preprocessing plan generated by decide_preprocessing_steps.

        Applies column dropping, missing value imputation, duplicate removal,
        and outlier handling based on the provided structured plan.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if plan is None:
            return {
                "status": "error",
                "message": "Plan is empty"
            }

        df = state.df.copy()
        actions = {
            "dropped_columns": [],
            "missing_filled": {},
            "duplicates_removed": 0,
            "outliers_handled": {}
        }

        for col in plan.get("drop_columns", []):
            if col in df.columns:
                df = df.drop(columns=[col])
                actions["dropped_columns"].append(col)

        fill_strategy = plan.get("fill_missing", {})

        for col, method in fill_strategy.items():
            if col not in df.columns:
                actions["missing_filled"][col] = "column_not_found"
                continue

            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                mode = df[col].mode()
                if mode.empty:
                    actions["missing_filled"][col] = "mode_not_available"
                    continue
                df[col] = df[col].fillna(mode[0])
            elif method == "drop":
                df = df.dropna(subset=[col])
            else:
                actions["missing_filled"][col] = "unsupported_method"
                continue

            actions["missing_filled"][col] = method

        if plan.get("drop_duplicates", False):
            rows_before = len(df)
            df = df.drop_duplicates()
            actions["duplicates_removed"] = rows_before - len(df)

        outlier_strategy = plan.get("handle_outliers", {})

        for col, method in outlier_strategy.items():
            if col not in df.columns:
                actions["outliers_handled"][col] = "column_not_found"
                continue

            if col not in df.select_dtypes(include="number").columns:
                actions["outliers_handled"][col] = "not_numeric"
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                actions["outliers_handled"][col] = "zero_iqr"
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if method == "drop":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == "cap":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "ignore":
                pass
            else:
                actions["outliers_handled"][col] = "unsupported_method"
                continue

            actions["outliers_handled"][col] = method

        state.df = df

        return {
            "status": "ok",
            "actions": actions,
            "shape_after": df.shape
        }

    @tool
    def generate_eda_report(target_col: str = None) -> dict:
        """Generate a concise EDA report for the current dataset.

        The report summarizes dataset structure, missing values, duplicates,
        outliers, target balance, and basic preprocessing recommendations.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df

        report = {
            "dataset_overview": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "missing_values": {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percent": (df.isnull().mean() * 100).round(2).to_dict()
            },
            "duplicates": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percent": round(df.duplicated().mean() * 100, 2)
            },
            "outliers": {},
            "target_balance": None,
            "recommendations": []
        }

        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            if col == target_col:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            report["outliers"][col] = {
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / len(df) * 100, 2)
                if len(df) > 0 else 0
            }

        if target_col is not None and target_col in df.columns:
            counts = df[target_col].value_counts(dropna=False)
            percents = df[target_col].value_counts(normalize=True, dropna=False) * 100

            report["target_balance"] = {
                "target_column": target_col,
                "class_counts": counts.to_dict(),
                "class_percent": percents.round(2).to_dict(),
                "is_imbalanced": bool(percents.max() > 70)
            }

        missing_percent = df.isnull().mean() * 100

        for col, percent in missing_percent.items():
            if percent > 50:
                report["recommendations"].append(
                    f"Consider dropping '{col}' because it has {percent:.2f}% missing values."
                )
            elif percent > 0:
                report["recommendations"].append(
                    f"Consider imputing missing values in '{col}'."
                )

        if report["duplicates"]["duplicate_rows"] > 0:
            report["recommendations"].append(
                "Consider removing duplicate rows."
            )

        for col, stats in report["outliers"].items():
            if stats["outlier_count"] > 0:
                report["recommendations"].append(
                    f"Consider handling outliers in '{col}'."
                )

        if report["target_balance"] is not None and report["target_balance"]["is_imbalanced"]:
            report["recommendations"].append(
                f"Target column '{target_col}' is imbalanced. Consider class weights, stratified split, oversampling, or undersampling."
            )

        return {
            "status": "ok",
            "report": report
        }

    @tool
    def export_clean_dataset(filename: str = "cleaned_dataset.csv") -> dict:
        """Export the current cleaned dataset to a CSV file.

        Saves the current state.df after EDA cleaning and preprocessing.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }
        path = f'./data/{filename}'
        state.df.to_csv(path, index=False)

        return {
            "status": "ok",
            "path": path,
            "shape": state.df.shape
        }



    return [drop_cols, analyze_missing, fill_missing, eda_summary, analyze_duplicates,
            drop_duplicates, analyze_outliers, handle_outliers, analyze_target_balance, apply_preprocessing_plan, generate_eda_report,
            export_clean_dataset, columns_description]


