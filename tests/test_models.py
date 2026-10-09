import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from src.models import (MODEL_NAMES, STRATEGIES, build_baseline, build_pipeline, make_preprocessor)
from src.preprocessing import TARGET, load_data, prepare, train_test


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = train_test(prepare(load_data()))
    # `cluster` factice (le vrai vient de 02_clustering.ipynb) : suffit pour tester la mécanique
    X_train = X_train.assign(cluster=np.arange(len(X_train)) % 4)
    X_test = X_test.assign(cluster=np.arange(len(X_test)) % 4)
    return X_train.iloc[:800], X_test.iloc[:300], y_train.iloc[:800], y_test.iloc[:300]


def test_toutes_les_combinaisons_se_construisent():
    for name in MODEL_NAMES:
        for strategy in STRATEGIES:
            pipe = build_pipeline(name, strategy)
            if name == "Gradient Boosting" and strategy == "class_weight":
                assert pipe is None            # combinaison volontairement absente
            else:
                assert pipe is not None


def test_smote_seulement_dans_la_strategie_smote():
    assert "smote" in build_pipeline("Logistic Regression", "smote").named_steps
    assert isinstance(build_pipeline("Logistic Regression", "smote").named_steps["smote"], SMOTE)
    assert "smote" not in build_pipeline("Logistic Regression", "none").named_steps
    assert "smote" not in build_pipeline("Logistic Regression", "class_weight").named_steps


def test_fit_predict_sans_churn_dans_x(data):
    X_train, X_test, y_train, y_test = data
    assert TARGET not in X_train.columns
    for strategy in STRATEGIES:
        pipe = build_pipeline("Logistic Regression", strategy).fit(X_train, y_train)
        assert pipe.predict(X_test).shape == (len(X_test),)


def test_cluster_et_feature_engineering_changent_les_colonnes(data):
    X_train, _, _, _ = data
    n = lambda uc, uf: make_preprocessor(uc, uf).fit_transform(X_train).shape[1]
    assert n(True, True) == n(False, True) + 4       # cluster : 4 colonnes one-hot
    assert n(False, False) < n(False, True)          # sans FE : moins de colonnes


def test_baseline_ne_predit_que_la_classe_majoritaire(data):
    X_train, X_test, y_train, _ = data
    pred = build_baseline().fit(X_train, y_train).predict(X_test)
    assert set(np.unique(pred)) == {0}


def test_cv_fold_scores_une_ligne_par_pli(data):
    from src.evaluate import cv_fold_scores, stratified_cv
    X_train, _, y_train, _ = data
    out = cv_fold_scores(build_pipeline("Logistic Regression", "none"), X_train, y_train, stratified_cv(3, 2))
    assert out.shape[0] == 6                                   # 3 plis x 2 répétitions
    assert {"roc_auc", "f1", "recall", "precision"} <= set(out.columns)
