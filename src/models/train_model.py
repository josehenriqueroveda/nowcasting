import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.externals import joblib

from src.features.build_features import build_features


def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)


def train_and_evaluate():
    config = load_config()
    X_train, X_test, y_train, y_test, scaler = build_features()

    tscv = TimeSeriesSplit(n_splits=3)
    model_lgb = lgb.LGBMClassifier(**config["model"]["lgbm"])

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

    y_proba = model_lgb.predict_proba(X_test)[:, 1]
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))

    joblib.dump(model_lgb, "models/trained/lgbm_model.pkl")
    joblib.dump(scaler, "models/trained/scaler.pkl")
