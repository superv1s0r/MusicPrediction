import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats

RANDOM_STATE = 42

# ---------- Load Data ----------
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

# ---------- Transformers ----------
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

# ---------- Preprocessing ----------
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

    pre = make_pipeline(
        FunctionTransformer(add_missing_flags, feature_names_out="one-to-one"),
        ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])
    )
    return pre

# ---------- Training ----------
def train_fast(df: pd.DataFrame, target: str):
    X, y = split_target(df, target)
    y = LabelEncoder().fit_transform(y)

    pre = build_preprocessor(X)
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    return pipe

# ---------- Run ----------
if __name__ == "__main__":
    df = load_data("./features_3_sec.csv")
    model = train_fast(df, target="label")


# --------- Different train, includes ROC curve, and confusion_matrix ------------
def train_fast_and_visualize(df: pd.DataFrame, target: str):
    X, y = split_target(df, target)

    # Ensure target is encoded consistently
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Transform target labels into numeric

    # Preprocessing and model pipeline
    pre = build_preprocessor(X)
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # Fit the pipeline
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred_encoded = pipe.predict(X_test)

    # Inverse transform predictions to original genre names
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

    # Inverse transform actual test labels to original genre names for reporting
    y_test_original = label_encoder.inverse_transform(y_test)

    # Classification Report
    print(classification_report(y_test_original, y_pred_original, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test_original, y_pred_original)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp_cm.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, pipe.predict_proba(X_test)[:, 1])
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title('ROC Curve')
    plt.show()

    return pipe, label_encoder
