"""
Preprocessing du projet churn (J3).

Principe anti-leakage :
- prepare()  : transformations SANS apprentissage (nettoyage, feature engineering).
               Peut être appliqué avant le split, et aussi dans Streamlit.
- build_preprocessor() : transformations AVEC apprentissage (imputation, scaling, encodage).
               À fitter UNIQUEMENT sur le train (idéalement dans un Pipeline).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "wa-fn-usec-telco-customer-churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "Churn"
RANDOM_STATE = 42

# Colonnes Yes/No (+ gender) -> 0/1
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CAT_COLS = ["Contract", "InternetService", "PaymentMethod", "tenure_group"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge", "n_services"]
PASS_COLS = BINARY_COLS + ["SeniorCitizen"]


def load_data(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage + Feature Engineering, sans apprentissage (pas de fit)."""
    df = df.copy()

    # 1) Identifiant inutile
    df = df.drop(columns=["customerID"], errors="ignore")

    # 2) TotalCharges : texte -> nombre (les cases vides deviennent NaN, imputées plus tard dans le Pipeline)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3) Modalités redondantes -> "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # 4) Binaires -> 0/1
    yes_no = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for c in BINARY_COLS:
        df[c] = df[c].map(yes_no).astype(int)

    # 5) Feature Engineering
    df["n_services"] = df[SERVICE_COLS].sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["avg_charge"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    # 6) Cible -> 0/1 (si présente)
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    """Sépare X et y (y = Churn). Churn ne reste JAMAIS dans X."""
    return df.drop(columns=[TARGET]), df[TARGET]


def train_test(df: pd.DataFrame, test_size=0.2, random_state=RANDOM_STATE):
    """Split stratifié, à faire AVANT tout fit."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Imputation + standardisation + one-hot. À fitter sur le train uniquement."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # TotalCharges manquant <=> tenure = 0
        ("scaler", StandardScaler()),
    ])
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        [
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
            ("bin", "passthrough", PASS_COLS),
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


if __name__ == "__main__":
    # Génère data/processed/train.csv et test.csv (version préparée, avant imputation/scaling)
    df = prepare(load_data())
    X_train, X_test, y_train, y_test = train_test(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)
    print("train :", X_train.shape, "| test :", X_test.shape)
    print("churn train : %.3f | churn test : %.3f" % (y_train.mean(), y_test.mean()))
