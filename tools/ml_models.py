from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "decision_tree_regressor": DecisionTreeRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    "svr": SVR,
    "knn_regressor": KNeighborsRegressor,

    "logistic_regression": LogisticRegression,
    "decision_tree_classifier": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "svc": SVC,
    "knn_classifier": KNeighborsClassifier,
}


REGRESSION_MODELS = {
    "linear_regression": {
        "description": "Simple baseline model for regression tasks.",
        "recommended_when": [
            "target variable is numeric",
            "relationship between features and target is expected to be close to linear",
            "fast and interpretable baseline is needed",
        ],
        "limitations": [
            "may perform poorly on complex nonlinear relationships",
            "sensitive to outliers",
        ],
    },
    "ridge": {
        "description": "Linear regression with L2 regularization.",
        "recommended_when": [
            "target variable is numeric",
            "dataset has many correlated features",
            "more stable linear model is needed",
        ],
        "limitations": [
            "may not capture complex nonlinear relationships",
        ],
    },
    "lasso": {
        "description": "Linear regression with L1 regularization.",
        "recommended_when": [
            "target variable is numeric",
            "feature selection may be useful",
            "dataset has many weak or irrelevant features",
        ],
        "limitations": [
            "may remove useful correlated features",
            "may not capture complex nonlinear relationships",
        ],
    },
    "decision_tree_regressor": {
        "description": "Tree-based model for regression tasks.",
        "recommended_when": [
            "target variable is numeric",
            "nonlinear relationships are expected",
            "model interpretability is important",
        ],
        "limitations": [
            "can overfit the training data",
            "usually less stable than ensemble models",
        ],
    },
    "random_forest_regressor": {
        "description": "Ensemble of decision trees for regression tasks.",
        "recommended_when": [
            "target variable is numeric",
            "dataset is tabular",
            "nonlinear relationships are expected",
            "strong general-purpose baseline is needed",
        ],
        "limitations": [
            "less interpretable than a single tree",
            "can be slower than simple linear models",
        ],
    },
    "gradient_boosting_regressor": {
        "description": "Boosting model based on sequential decision trees for regression tasks.",
        "recommended_when": [
            "target variable is numeric",
            "high predictive quality is important",
            "nonlinear relationships are expected",
        ],
        "limitations": [
            "requires more careful hyperparameter tuning",
            "can overfit if not configured properly",
        ],
    },
    "svr": {
        "description": "Support Vector Regression model.",
        "recommended_when": [
            "target variable is numeric",
            "dataset is small or medium-sized",
            "features are properly scaled",
        ],
        "limitations": [
            "does not scale well to large datasets",
            "sensitive to feature scaling",
        ],
    },
    "knn_regressor": {
        "description": "K-nearest neighbors model for regression tasks.",
        "recommended_when": [
            "target variable is numeric",
            "local similarity between objects is important",
            "simple non-parametric baseline is needed",
        ],
        "limitations": [
            "sensitive to feature scaling",
            "can be slow on large datasets",
            "may perform poorly with many features",
        ],
    },
}


CLASSIFICATION_MODELS = {
    "logistic_regression": {
        "description": "Simple baseline model for classification tasks.",
        "recommended_when": [
            "target variable is categorical",
            "binary or multiclass classification is needed",
            "fast and interpretable baseline is needed",
        ],
        "limitations": [
            "may perform poorly on complex nonlinear relationships",
            "sensitive to feature scaling",
        ],
    },
    "decision_tree_classifier": {
        "description": "Tree-based model for classification tasks.",
        "recommended_when": [
            "target variable is categorical",
            "nonlinear relationships are expected",
            "model interpretability is important",
        ],
        "limitations": [
            "can overfit the training data",
            "usually less stable than ensemble models",
        ],
    },
    "random_forest_classifier": {
        "description": "Ensemble of decision trees for classification tasks.",
        "recommended_when": [
            "target variable is categorical",
            "dataset is tabular",
            "nonlinear relationships are expected",
            "strong general-purpose baseline is needed",
        ],
        "limitations": [
            "less interpretable than a single tree",
            "can be slower than simple models",
        ],
    },
    "gradient_boosting_classifier": {
        "description": "Boosting model based on sequential decision trees for classification tasks.",
        "recommended_when": [
            "target variable is categorical",
            "high predictive quality is important",
            "nonlinear relationships are expected",
        ],
        "limitations": [
            "requires more careful hyperparameter tuning",
            "can overfit if not configured properly",
        ],
    },
    "svc": {
        "description": "Support Vector Classifier model.",
        "recommended_when": [
            "target variable is categorical",
            "dataset is small or medium-sized",
            "features are properly scaled",
            "complex decision boundary is expected",
        ],
        "limitations": [
            "does not scale well to large datasets",
            "sensitive to feature scaling",
        ],
    },
    "knn_classifier": {
        "description": "K-nearest neighbors model for classification tasks.",
        "recommended_when": [
            "target variable is categorical",
            "local similarity between objects is important",
            "simple non-parametric baseline is needed",
        ],
        "limitations": [
            "sensitive to feature scaling",
            "can be slow on large datasets",
            "may perform poorly with many features",
        ],
    },
}


REGRESSION_METRICS = {
    "mae": {
        "description": "Mean Absolute Error. Lower is better.",
        "recommended_when": [
            "error should be easy to interpret in target units",
            "large errors should not be punished too aggressively",
        ],
    },
    "mse": {
        "description": "Mean Squared Error. Lower is better.",
        "recommended_when": [
            "large errors should be penalized more strongly",
        ],
    },
    "rmse": {
        "description": "Root Mean Squared Error. Lower is better.",
        "recommended_when": [
            "error should be interpreted in the same units as the target",
            "large errors should be penalized more strongly",
        ],
    },
    "r2": {
        "description": "R-squared score. Higher is better.",
        "recommended_when": [
            "need to estimate how much target variance is explained by the model",
        ],
    },
}


CLASSIFICATION_METRICS = {
    "accuracy": {
        "description": "Share of correctly classified objects. Higher is better.",
        "recommended_when": [
            "classes are approximately balanced",
            "all types of errors are equally important",
        ],
    },
    "precision": {
        "description": "Share of true positive predictions among all positive predictions. Higher is better.",
        "recommended_when": [
            "false positives are especially undesirable",
        ],
    },
    "recall": {
        "description": "Share of real positive objects found by the model. Higher is better.",
        "recommended_when": [
            "false negatives are especially undesirable",
        ],
    },
    "f1": {
        "description": "Harmonic mean of precision and recall. Higher is better.",
        "recommended_when": [
            "classes are imbalanced",
            "both false positives and false negatives matter",
        ],
    },
    "roc_auc": {
        "description": "Ability of the model to rank positive objects above negative objects. Higher is better.",
        "recommended_when": [
            "binary classification is used",
            "model can return probabilities or decision scores",
        ],
    },
}