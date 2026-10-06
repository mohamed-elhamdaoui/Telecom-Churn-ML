import numpy as np
from src.preprocessing import (TARGET, NUM_COLS, build_preprocessor, load_data,
                               prepare, train_test)


def _data():
    df = prepare(load_data())
    return train_test(df)


def test_prepare_basique():
    df = prepare(load_data())
    assert "customerID" not in df.columns
    assert df.shape[0] == 7043
    assert df["TotalCharges"].isna().sum() == 11          # NaN gardés, imputés plus tard
    assert set(df[TARGET].unique()) == {0, 1}
    assert not df.isin(["No internet service", "No phone service"]).any().any()


def test_pas_de_churn_dans_X():
    X_train, X_test, _, _ = _data()
    assert TARGET not in X_train.columns and TARGET not in X_test.columns


def test_split_stratifie():
    _, _, y_train, y_test = _data()
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_pipeline_sans_nan_et_sans_leakage():
    X_train, X_test, _, _ = _data()
    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)         # fit sur le train seulement
    Xt_test = pre.transform(X_test)
    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert not Xt_train.isna().any().any() and not Xt_test.isna().any().any()
    # scaler ajusté sur le train => moyenne train ~ 0, moyenne test != 0 exactement
    assert np.allclose(Xt_train[NUM_COLS].mean(), 0, atol=1e-8)
    assert not np.allclose(Xt_test[NUM_COLS].mean(), 0, atol=1e-8)
    assert TARGET not in Xt_train.columns