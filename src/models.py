"""
Modèles et Pipelines de classification (J6-J7).

Tout ce qui apprend (imputation, scaling, encodage, SMOTE, modèle) est dans un Pipeline :
il est donc ajusté uniquement sur les données d'entraînement (pas de data leakage).
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import CAT_COLS, NUM_COLS, PASS_COLS, RANDOM_STATE, TARGET

FE_NUM = ["avg_charge", "n_services"]      # variables numériques créées par Feature Engineering
FE_CAT = ["tenure_group"]                  # variable catégorielle créée par Feature Engineering

STRATEGIES = ["none", "class_weight", "smote"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Gradient Boosting"]


def load_split(processed_dir):
    """Charge train/test avec la colonne `cluster` (produits par 02_clustering.ipynb)."""
    train = pd.read_csv(f"{processed_dir}/train_with_clusters.csv")
    test = pd.read_csv(f"{processed_dir}/test_with_clusters.csv")
    return (train.drop(columns=[TARGET]), test.drop(columns=[TARGET]),
            train[TARGET], test[TARGET])


def make_preprocessor(use_cluster=False, use_fe=True):
    """ColumnTransformer paramétrable : avec/sans `cluster`, avec/sans Feature Engineering."""
    num = [c for c in NUM_COLS if use_fe or c not in FE_NUM]
    cat = [c for c in CAT_COLS if use_fe or c not in FE_CAT]
    if use_cluster:
        cat = cat + ["cluster"]            # le cluster est une catégorie (pas une quantité)
    return ColumnTransformer(
        [("num", SkPipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                             ("scaler", StandardScaler())]), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
         ("bin", "passthrough", PASS_COLS)],
        verbose_feature_names_out=False,
    )


def make_model(name, strategy="none"):
    """Retourne le modèle, ou None si la combinaison n'existe pas (ex. Gradient Boosting + class_weight)."""
    cw = "balanced" if strategy == "class_weight" else None
    rs = RANDOM_STATE
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=rs)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, class_weight=cw, random_state=rs)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=rs, n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight=cw, random_state=rs)
    if name == "Gradient Boosting":
        if cw is not None:
            return None                    # GradientBoostingClassifier n'a pas de paramètre class_weight
        return GradientBoostingClassifier(random_state=rs)
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(name, strategy="none", use_cluster=False, use_fe=True):
    """preprocessor -> (SMOTE) -> modèle. SMOTE n'agit qu'à l'entraînement (jamais au predict/test)."""
    model = make_model(name, strategy)
    if model is None:
        return None
    steps = [("pre", make_preprocessor(use_cluster, use_fe))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def build_baseline():
    return Pipeline([("pre", make_preprocessor()),
                     ("model", DummyClassifier(strategy="most_frequent"))])
