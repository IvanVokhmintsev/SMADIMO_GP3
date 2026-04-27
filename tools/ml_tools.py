from typing import List

from langchain.tools import tool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import os
import joblib
import time

from tools.ml_models import (
    MODEL_REGISTRY,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
)

metrics_dict = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2": r2_score,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}

def make_ml_tools(state):

    @tool
    def get_model_catalog(task_type: str) -> dict:
        """Return available machine learning models for the selected task type.

        Provides a list of candidate models that can be used for training.
        Each model includes a description, recommended usage scenarios,
        and known limitations. This helps the agent choose appropriate models
        based on dataset characteristics.

        Args:
            task_type: Type of machine learning task.
                       Supported values:
                       - 'regression': models for numeric target prediction
                       - 'classification': models for categorical target prediction
        """
        if task_type == "regression":
            return {
                "status": "ok",
                "task_type": task_type,
                "models": REGRESSION_MODELS
            }

        if task_type == "classification":
            return {
                "status": "ok",
                "task_type": task_type,
                "models": CLASSIFICATION_MODELS
            }

        return {
            "status": "error",
            "message": "task_type must be 'regression' or 'classification'"
        }

    @tool
    def get_metric_catalog(task_type: str) -> dict:
        """Return available evaluation metrics for the selected task type.

        Provides a list of evaluation metrics that can be used to assess
        model performance. Each metric includes a description and guidance
        on when it should be used.

        Args:
            task_type: Type of machine learning task.
                       Supported values:
                       - 'regression': metrics for numeric prediction
                       - 'classification': metrics for categorical prediction
        """
        if task_type == "regression":
            return {
                "status": "ok",
                "task_type": task_type,
                "metrics": REGRESSION_METRICS
            }

        if task_type == "classification":
            return {
                "status": "ok",
                "task_type": task_type,
                "metrics": CLASSIFICATION_METRICS
            }

        return {
            "status": "error",
            "message": "task_type must be 'regression' or 'classification'"
        }
    @tool
    def train_and_evaluate_models(
        target_col: str,
        task_type: str,
        model_names: List[str],
        metric_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """Train, save, and evaluate selected models on the dataset.

        Splits the dataset into train and test parts, trains the selected models,
        saves trained models to state.models and as .pkl files, and evaluates them using the
        chosen metrics.

        Args:
            target_col: Name of the target column.
            task_type: Type of machine learning task.
                       Supported values:
                       - 'regression': train regression models
                       - 'classification': train classification models
            model_names: List of model names to train.
            metric_names: List of metrics to evaluate.
            test_size: Proportion of test data.
            random_state: Random seed for reproducibility.
        """

        if state.df is None:
            return {
                "status": "error",
                "message": "DataFrame is empty"
            }

        if target_col not in state.df.columns:
            return {
                "status": "error",
                "message": "Target column not found"
            }

        if task_type not in ["regression", "classification"]:
            return {
                "status": "error",
                "message": "task_type must be 'regression' or 'classification'"
            }

        df = state.df.dropna().copy()

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        if not hasattr(state, "models"):
            state.models = {}

        results = {}

        os.makedirs("saved_models", exist_ok=True)

        for model_name in model_names:

            if model_name not in MODEL_REGISTRY:
                results[model_name] = {
                    "status": "error",
                    "message": "Model not found in MODEL_REGISTRY"
                }
                continue

            model_class = MODEL_REGISTRY[model_name]
            model = model_class()

            model.fit(X_train, y_train)

            state.models[model_name] = model
            model_path = f"saved_models/{model_name}_{int(time.time())}.pkl"
            joblib.dump(model, model_path)

            y_pred = model.predict(X_test)

            model_result = {}

            for metric in metric_names:
                model_result[metric] = metrics_dict.get(metric)(y_test, y_pred)

            results[model_name] = {
                "status": "ok",
                "metrics": model_result,
                "model_path": model_path
            }

        return {
            "status": "ok",
            "saved_models": list(state.models.keys()),
            "results": results
        }


    return [get_model_catalog, get_metric_catalog, train_and_evaluate_models]