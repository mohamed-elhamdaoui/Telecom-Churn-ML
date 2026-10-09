"""Évaluation des modèles : validation croisée, métriques sur le test, graphiques."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate

from src.preprocessing import RANDOM_STATE

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def stratified_cv(n_splits=5, n_repeats=1):
    if n_repeats == 1:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)


def cv_scores(pipeline, X, y, cv):
    """Moyenne et écart-type des métriques sur les plis (classe positive = churn)."""
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out = {}
    for m in SCORING:
        out[m] = res[f"test_{m}"].mean()
        out[f"{m}_std"] = res[f"test_{m}"].std()
    return out


def test_metrics(pipeline, X_test, y_test):
    """Métriques sur le jeu de test (pipeline déjà ajusté sur le train)."""
    pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        score = pipeline.predict_proba(X_test)[:, 1]
    else:
        score = pipeline.decision_function(X_test)
    return {"accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, score)}


def plot_confusions(fitted, X_test, y_test, ncols=2):
    """Matrices de confusion pour un dict {nom: pipeline ajusté}."""
    n = len(fitted)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    for ax, (name, pipe) in zip(np.atleast_1d(axes).ravel(), fitted.items()):
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, display_labels=["Reste", "Churn"],
                                              cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_roc(fitted, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set_title("Courbes ROC (test)")
    ax.legend(loc="lower right")
    plt.show()


def cv_fold_scores(pipeline, X, y, cv):
    """Scores pli par pli (DataFrame : une ligne par pli, une colonne par métrique).
    Avec le même `cv` (même random_state), les plis sont identiques d'une configuration à l'autre :
    on peut donc comparer deux configurations pli par pli (comparaison appariée)."""
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
