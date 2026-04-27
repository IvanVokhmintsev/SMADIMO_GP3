from langchain.tools import tool

from typing import Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def make_fe_tools(state):

    @tool
    def create_ratio_feature(input: dict) -> dict:
        """Create a new feature as a ratio of two numeric columns.

        Args:
            input: Dictionary containing:
                - numerator: name of numerator column
                - denominator: name of denominator column
                - new_column: name of new feature
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }
        num = input["numerator"]
        den = input["denominator"]
        new_col = input["new_column"]


        if not hasattr(state, "features"):
            state.features = []

        df = state.df

        if num not in df.columns or den not in df.columns:
            return {"status": "error", "message": "Column not found"}

        df[new_col] = df[num] / (df[den] + 1e-9)

        state.df = df
        state.features.append(new_col)

        return {"status": "ok", "feature": new_col}

    @tool
    def create_difference_feature(input: dict) -> dict:
        """Create a new feature as the difference between two numeric columns.

        Args:
            input: Dictionary containing:
                - column_a: first numeric column
                - column_b: second numeric column
                - new_column: name of new feature
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        col_a = input["column_a"]
        col_b = input["column_b"]
        new_col = input["new_column"]

        df = state.df

        if col_a not in df.columns or col_b not in df.columns:
            return {
                "status": "error",
                "message": "Column not found"
            }

        df[new_col] = df[col_a] - df[col_b]

        state.df = df
        state.features.append(new_col)

        return {
            "status": "ok",
            "feature": new_col
        }

    @tool
    def scale_numeric_features(input: dict) -> dict:
        """Scale selected numeric features using a chosen scaling method.

        Args:
            input: Dictionary containing:
                - columns: list of numeric columns to scale
                - method: scaling method, either 'standard' or 'minmax'
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        columns = input["columns"]
        method = input.get("method", "standard")

        df = state.df.copy()

        missing_cols = [col for col in columns if col not in df.columns]

        if missing_cols:
            return {
                "status": "error",
                "message": "Some columns were not found",
                "missing_columns": missing_cols
            }

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            return {
                "status": "error",
                "message": "Unsupported scaling method"
            }

        df[columns] = scaler.fit_transform(df[columns])

        state.df = df

        return {
            "status": "ok",
            "scaled_columns": columns,
            "method": method
        }

    @tool
    def decide_feature_engineering(target_col: str = None) -> dict:
        """Decide which feature engineering steps to apply.

        Suggests at least two feature transformations based on dataset structure.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        df = state.df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        plan = {
            "create_ratio": {},
            "create_difference": {},
            "scale": [],
            "notes": []
        }

        # Try to pick 2 numeric columns for feature creation
        if len(numeric_cols) >= 2:
            col_a = numeric_cols[0]
            col_b = numeric_cols[1]

            plan["create_ratio"] = {"numerator": col_a, "denominator": col_b, "new_column": f"{col_a}_to_{col_b}_ratio"}

            plan["notes"].append(
                f"Using '{col_a}' and '{col_b}' to generate ratio and difference features."
            )

        # Suggest scaling for numeric columns (excluding target)
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if numeric_cols:
            plan["scale"] = numeric_cols
            plan["notes"].append(
                "Scaling numeric features using standard scaling."
            )

        return {
            "status": "ok",
            "plan": plan
        }

    @tool
    def apply_feature_engineering_plan(plan: Dict) -> dict:
        """Apply a feature engineering plan.

        Creates ratio and difference features and optionally scales numeric columns.
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

        if not hasattr(state, "features"):
            state.features = []

        df = state.df.copy()
        created_features = []
        actions = {}

        ratio_plan = plan.get("create_ratio", {})

        if ratio_plan:
            num = ratio_plan["numerator"]
            den = ratio_plan["denominator"]
            new_col = ratio_plan["new_column"]

            if num in df.columns and den in df.columns:
                df[new_col] = df[num] / (df[den] + 1e-9)
                created_features.append(new_col)
                actions["create_ratio"] = new_col
            else:
                actions["create_ratio"] = "column_not_found"

        difference_plan = plan.get("create_difference", {})

        if difference_plan:
            col_a = difference_plan["column_a"]
            col_b = difference_plan["column_b"]
            new_col = difference_plan["new_column"]

            if col_a in df.columns and col_b in df.columns:
                df[new_col] = df[col_a] - df[col_b]
                created_features.append(new_col)
                actions["create_difference"] = new_col
            else:
                actions["create_difference"] = "column_not_found"

        scale_columns = plan.get("scale", [])

        if scale_columns:
            existing_scale_columns = [
                col for col in scale_columns
                if col in df.columns
            ]

            if existing_scale_columns:
                scaler = StandardScaler()
                df[existing_scale_columns] = scaler.fit_transform(
                    df[existing_scale_columns]
                )
                actions["scale"] = existing_scale_columns
            else:
                actions["scale"] = "no_valid_columns"

        state.df = df
        state.features.extend(created_features)

        return {
            "status": "ok",
            "actions": actions,
            "created_features": created_features,
            "shape_after": df.shape
        }

    @tool
    def generate_fe_report() -> dict:
        """Generate a feature engineering report.

        Returns created features, dataset shape, numeric columns,
        and basic information about the current feature engineering state.
        """
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if not hasattr(state, "features"):
            state.features = []

        df = state.df

        return {
            "status": "ok",
            "created_features": state.features,
            "created_features_count": len(state.features),
            "shape": df.shape,
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "columns": list(df.columns)
        }
    @tool
    def export_engineered_dataset(path: str = "engineered_dataset.csv") -> dict:
        """Export the current feature-engineered dataset to a CSV file."""
        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        state.df.to_csv('./data/' + path, index=False)

        return {
            "status": "ok",
            "path": path,
            "shape": state.df.shape
        }



    return [
        create_ratio_feature,
        create_difference_feature,
        scale_numeric_features,
        decide_feature_engineering,
        apply_feature_engineering_plan,
        generate_fe_report,
        export_engineered_dataset
    ]
