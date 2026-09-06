"""Logistic baseline + XGBoost, light val tuning, probability calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.features import FeatureEncoder

# Regularized configs. n_estimators is an upper bound — early stopping on the
# validation set picks the real tree count, so depth/regularization vary here.
XGB_GRID = [
    {"max_depth": 3, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "reg_alpha": 0.0, "gamma": 0.5},
    {"max_depth": 5, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 3.0, "reg_alpha": 0.5, "gamma": 1.0},
    {"max_depth": 4, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "reg_alpha": 1.0, "gamma": 1.0},
    {"max_depth": 3, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0, "reg_alpha": 0.0, "gamma": 0.3},
]

XGB_MAX_ROUNDS = 600
XGB_EARLY_STOPPING = 40
XGB_LEARNING_RATE = 0.05


def scale_pos_weight(y: np.ndarray) -> float:
    """n_neg / n_pos on the actual training prevalence. No SMOTE."""
    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    return float(n_neg / n_pos)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def fit_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Pipeline, pd.DataFrame]:
    clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    lr = clf.named_steps["lr"]
    coefs = pd.DataFrame({"feature": list(X_train.columns), "coef": lr.coef_.ravel()})
    coefs["abs"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs", ascending=False)
    return clf, coefs


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """Grid over regularized configs, early-stopped on val.

    Model selection maximizes validation **average precision** (the headline PR
    metric we report), with lower logloss as a tie-breaker — not raw logloss.
    Early stopping on val AUC-PR sets the tree count, so we do not overfit the
    festive test set. No SMOTE; class imbalance handled via scale_pos_weight only.
    """
    spw = scale_pos_weight(y_train)
    best_model = None
    best_key: tuple[float, float] | None = None
    best_params: dict[str, Any] = {}
    best_scores: dict[str, Any] = {}
    for params in XGB_GRID:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            tree_method="hist",
            learning_rate=XGB_LEARNING_RATE,
            n_estimators=XGB_MAX_ROUNDS,
            early_stopping_rounds=XGB_EARLY_STOPPING,
            random_state=RANDOM_SEED,
            n_jobs=4,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p = model.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, p))
        loss = float(log_loss(y_val, p, labels=[0, 1]))
        key = (ap, -loss)  # higher AP wins; lower logloss breaks ties
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_params = dict(params)
            best_scores = {
                "val_average_precision": ap,
                "val_logloss": loss,
                "best_iteration": int(getattr(model, "best_iteration", -1)),
            }
    assert best_model is not None
    meta = {
        "best_params": best_params,
        "selection_metric": "val_average_precision",
        "scale_pos_weight": spw,
        "n_train": int(len(y_train)),
        "train_prevalence": float(np.mean(y_train)),
        **best_scores,
    }
    return best_model, meta


def calibrate_probabilities(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_ref: pd.DataFrame | None = None,
    y_ref: np.ndarray | None = None,
) -> tuple[Any, str, dict[str, float]]:
    """Platt (sigmoid) first; isotonic if val is large and ECE still off."""
    y_val = np.asarray(y_val).astype(int)
    p_raw = model.predict_proba(X_val)[:, 1]
    ece_raw = expected_calibration_error(y_val, p_raw)

    try:
        from sklearn.frozen import FrozenEstimator

        frozen = FrozenEstimator(model)
        platt = CalibratedClassifierCV(frozen, method="sigmoid")
        platt.fit(X_val, y_val)
    except Exception:
        platt = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        platt.fit(X_val, y_val)
    p_platt = platt.predict_proba(X_val)[:, 1]
    ece_platt = expected_calibration_error(y_val, p_platt)
    chosen: Any = platt
    method = "platt_sigmoid"
    ece_iso = None

    if len(y_val) >= 2000 and ece_platt > 0.03:
        try:
            from sklearn.frozen import FrozenEstimator

            iso = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        except Exception:
            iso = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        iso.fit(X_val, y_val)
        p_iso = iso.predict_proba(X_val)[:, 1]
        ece_iso = expected_calibration_error(y_val, p_iso)
        if ece_iso < ece_platt:
            chosen = iso
            method = "isotonic"
    meta = {
        "ece_raw": ece_raw,
        "ece_platt": ece_platt,
        "ece_isotonic": ece_iso,
        "method": method,
    }
    return chosen, method, meta


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def reliability_points(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos
