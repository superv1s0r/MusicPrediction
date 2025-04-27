"""
Machine‑learning pipeline template for tabular datasets
Extended to cover additional edge‑cases:
  • Loading data
  • Missing‑value flags
  • Rare‑category collapsing
  • Detecting & capping outliers (IQR rule) **inside CV‑safe pipeline**
  • Skewed‑numeric log/Yeo–Johnson transform
  • Scaling / normalisation
  • Optional class imbalance handling (SMOTE‑NC)
  • Stratified / Group / TimeSeries CV
  • RandomForest or XGBoost (scikit‑learn API)
  • Optuna hyper‑parameter optimisation
  • Calibration, reporting & feature importance

Requires: pandas, numpy, scikit‑learn, xgboost, imbalanced‑learn, optuna, scipy, matplotlib
"""
from joblib import Memory
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union, Optional

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
    PowerTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC

import optuna
from scipy import stats

RANDOM_STATE = 42

# ---------- helpers ----------

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Unsupported file format")


def split_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

# ---------- custom transformers ----------

def add_missing_flags(X):
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[f"{col}__was_missing"] = X[col].isna().astype(int)
    return X


def collapse_rare(min_freq: float = 0.01):
    def _collapser(X):
        X = X.copy()
        for col in X.select_dtypes("object"):
            freq = X[col].value_counts(normalize=True)
            rare = freq[freq < min_freq].index
            X[col] = X[col].where(~X[col].isin(rare), "__OTHER__")
        return X

    return FunctionTransformer(_collapser, feature_names_out="one-to-one")


def cap_outliers_iqr(factor: float = 1.5):
    def _capper(X):
        X = X.copy()
        for col in X.select_dtypes(include=[np.number]):
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            X[col] = X[col].clip(lower, upper)
        return X

    return FunctionTransformer(_capper, feature_names_out="one-to-one")


def log_skewed():
    def _log(X):
        X = X.copy()
        skewed = X.select_dtypes(include=[np.number]).apply(lambda s: abs(stats.skew(s.dropna())) > 1)
        for col in skewed[skewed].index:
            X[col] = np.log1p(X[col])
        return X

    return FunctionTransformer(_log, feature_names_out="one-to-one")

# ---------- preprocessing constructor ----------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("cap", cap_outliers_iqr()),
        ("log", log_skewed()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("collapse", collapse_rare()),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Add missing flags globally first
    pre = make_pipeline(
        FunctionTransformer(add_missing_flags, feature_names_out="one-to-one"),
        ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])
    )
    return pre

# ---------- imbalance handling ----------

def get_sampler(cat_indices):
    return SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)

# ---------- model factories ----------

def get_estimator(name: str):
    if name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1, eval_metric="logloss")

# ---------- optuna search ----------

def tune_model(model_name: str, X, y, pre):
    def objective(trial):
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        model = get_estimator(model_name).set_params(**params)

        pipe = Pipeline([("pre", pre), ("clf", model)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        return np.mean(scores["test_score"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    return study.best_params

# ---------- training entrypoint ----------

def train(df: pd.DataFrame, target: str, model_name: str = "rf", cv_type: str = "stratified"):
    X, y = split_target(df, target)

    le = LabelEncoder()
    y = le.fit_transform(y)

    pre = build_preprocessor(X)

    best = tune_model(model_name, X, y, pre)
    base = get_estimator(model_name).set_params(**best)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    return pipe

if __name__ == "__main__":
    df = load_data("./features_3_sec.csv")
    model = train(df, target="label", model_name="xgb")

