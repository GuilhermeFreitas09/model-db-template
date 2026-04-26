import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from simple_quant import config
from simple_quant.features import infer_horizon_from_target, split_train_val_test


def find_best_fbeta_threshold(y_true, probs, beta: float = 0.5) -> tuple[float, float]:
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = [fbeta_score(y_true, (probs > t).astype(int), beta=beta, zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def train_model(
    df_model: pd.DataFrame,
    target_col: str,
    train_percent: float = 0.6,
    val_percent: float = 0.2,
    n_iter: int = 20,
) -> tuple[XGBClassifier, dict, list[str]]:
    horizon = infer_horizon_from_target(target_col)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train_df, val_df, test_df = split_train_val_test(
        df_model=df_model,
        target_col=target_col,
        train_percent=train_percent,
        val_percent=val_percent,
        horizon=horizon,
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )
    
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    
    # Métricas para validação
    val_probs = model.predict_proba(X_val)[:, 1]
    val_threshold, val_best_fbeta = find_best_fbeta_threshold(y_val, val_probs, beta=0.5)
    val_preds = (val_probs > val_threshold).astype(int)
    
    # Métricas para teste
    test_probs = model.predict_proba(X_test)[:, 1]
    test_threshold, test_best_fbeta = find_best_fbeta_threshold(y_test, test_probs, beta=0.5)
    test_preds = (test_probs > test_threshold).astype(int)

    metrics = {
        "target_col": target_col,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "cv_best_score": float(random_search.best_score_),
        "val_auc_roc": float(roc_auc_score(y_val, val_probs)),
        "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "val_fbeta_0_5": float(val_best_fbeta),
        "val_threshold": float(val_threshold),
        "test_auc_roc": float(roc_auc_score(y_test, test_probs)),
        "test_precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "test_fbeta_0_5": float(test_best_fbeta),
        "test_threshold": float(test_threshold),
        "best_params": random_search.best_params_,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_period": {
            "start": str(train_df['data'].min()),
            "end": str(train_df['data'].max())
        },
        "val_period": {
            "start": str(val_df['data'].min()),
            "end": str(val_df['data'].max())
        },
        "test_period": {
            "start": str(test_df['data'].min()),
            "end": str(test_df['data'].max())
        },
    }
    return model, metrics, feature_cols


def save_model_artifacts(
    model,
    model_name: str,
    ticker: str,
    target_col: str,
    target_price_col: str,
    feature_cols: list[str],
    metrics: dict,
    base_dir: Optional[Path] = None,
) -> Path:
    base_dir = Path(base_dir or config.MODELS_DIR)
    artifact_dir = base_dir / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "ticker": ticker,
        "target_col": target_col,
        "target_price_col": target_price_col,
        "feature_cols": feature_cols,
        "threshold": metrics["val_threshold"],
        "created_at": datetime.now().isoformat(),
    }

    with open(artifact_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    return artifact_dir


def load_model_artifacts(artifact_path: str | Path) -> tuple[object, dict]:
    artifact_dir = Path(artifact_path)
    if artifact_dir.is_file():
        artifact_dir = artifact_dir.parent

    with open(artifact_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifact_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def score_row(model, X_row: pd.DataFrame, threshold: float) -> dict:
    probability = float(model.predict_proba(X_row)[0, 1])
    prediction = int(probability > threshold)
    return {"probability": probability, "prediction": prediction}
