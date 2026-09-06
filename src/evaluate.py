"""Test-set evaluation: PR, calibration, cost-optimal threshold, fairness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    ARTIFACTS,
    C_FN_INR,
    CONVERSION_LOSS_IF_PREPAID,
    EVAL_A_C_FN,
    EVAL_A_C_FP,
    MARGIN_RATE,
    VERIFY_COST_INR,
    VERIFY_RTO_REDUCTION,
)
from src.models import expected_calibration_error

plt.switch_backend("Agg")


def classification_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
    }


def pr_payload(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    prec, rec, thr = precision_recall_curve(y, p)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "thresholds": thr.tolist(),
    }


def fp_cost_per_order(
    amount_inr: np.ndarray,
    margin_rate: float = MARGIN_RATE,
    conversion_loss: float = CONVERSION_LOSS_IF_PREPAID,
) -> np.ndarray:
    return margin_rate * np.asarray(amount_inr, dtype=float) * conversion_loss


def total_cost(
    y: np.ndarray,
    p: np.ndarray,
    t: float,
    c_fn: float,
    c_fp: np.ndarray,
    fp_scale: float = 1.0,
) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= t
    fn = (y == 1) & ~pred
    fp = (y == 0) & pred
    return float(fn.sum() * c_fn + fp_scale * np.asarray(c_fp)[fp].sum())


def cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    c_fn: float,
    c_fp: np.ndarray,
    fp_scale: float = 1.0,
    n_grid: int = 61,
) -> pd.DataFrame:
    lo, hi = float(np.min(p)), float(np.max(p))
    ts = np.unique(
        np.concatenate(
            [
                [lo],
                np.quantile(p, np.linspace(0.0, 1.0, n_grid)),
                [min(0.99, hi + 1e-6)],
            ]
        )
    )
    rows = []
    for t in ts:
        pred = p >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        cost = total_cost(y, p, t, c_fn, c_fp, fp_scale=fp_scale)
        rows.append({"threshold": float(t), "cost": cost, "fn": fn, "fp": fp})
    return pd.DataFrame(rows).sort_values("threshold")


def pick_cost_optimal_threshold(
    y: np.ndarray,
    p: np.ndarray,
    c_fn: float,
    c_fp: np.ndarray,
    fp_scale: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(y, p, c_fn, c_fp, fp_scale=fp_scale)
    i = int(curve["cost"].idxmin())
    return float(curve.loc[i, "threshold"]), curve


def verification_cost(
    y: np.ndarray,
    p: np.ndarray,
    t_med: float,
    t_star: float,
    c_fn: float,
    c_fp: np.ndarray,
    verify_cost: float = VERIFY_COST_INR,
    rto_cut: float = VERIFY_RTO_REDUCTION,
) -> dict[str, float]:
    """Optional: medium band pays ₹verify_cost and cuts RTO among those flagged by ~25% (assumption)."""
    y = np.asarray(y).astype(int)
    high = p >= t_star
    med = (p >= t_med) & (p < t_star)
    low = p < t_med
    # High: treat as prepaid-required → FP pays c_fp, FN among high is 0 (forced prepaid, no COD RTO)
    # Remaining COD RTO: lows that are positive (missed) + medium positives not caught by verification
    fn_low = int(((y == 1) & low).sum())
    med_pos = int(((y == 1) & med).sum())
    med_neg = int(((y == 0) & med).sum())
    high_fp = int(((y == 0) & high).sum())
    # Verification reduces medium RTO by rto_cut; remaining still FN-like loss
    cost = (
        fn_low * c_fn
        + med_pos * (1.0 - rto_cut) * c_fn
        + med.sum() * verify_cost
        + np.asarray(c_fp)[(y == 0) & high].sum()
    )
    return {
        "expected_cost_with_verify_step": float(cost),
        "n_medium": int(med.sum()),
        "n_high": int(high.sum()),
        "verify_cost_inr": verify_cost,
        "assumed_rto_reduction_on_medium": rto_cut,
        "fn_low": fn_low,
        "med_pos": med_pos,
        "med_neg": med_neg,
        "high_fp": high_fp,
        "note": "Verification lift is an assumption, not a measured causal effect.",
    }


def fp_rate_by_group(y: np.ndarray, pred: np.ndarray, group: pd.Series) -> dict[str, dict[str, float]]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    group = pd.Series(group).reset_index(drop=True)
    out: dict[str, dict[str, float]] = {}
    for g, idx in group.groupby(group).groups.items():
        mask = np.zeros(len(y), dtype=bool)
        mask[list(idx)] = True
        neg = y == 0
        n_neg = int((mask & neg).sum())
        fp = int((mask & neg & (pred == 1)).sum())
        out[str(g)] = {
            "n": int(mask.sum()),
            "n_neg": n_neg,
            "fp": fp,
            "fp_rate": float(fp / n_neg) if n_neg else float("nan"),
            "flag_rate": float(pred[mask].mean()) if mask.any() else float("nan"),
        }
    return out


def save_pr_plot(y: np.ndarray, p: np.ndarray, path: Path, title: str) -> None:
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(rec, prec, color="#1f4e79", lw=2, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_calibration_plot(y: np.ndarray, p: np.ndarray, path: Path, title: str) -> None:
    from sklearn.calibration import calibration_curve

    mp, fp = calibration_curve(y, p, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(mp, fp, "o-", color="#c0392b", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_cost_plot(curve: pd.DataFrame, t_star: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(curve["threshold"], curve["cost"], color="#1f4e79", lw=2)
    ax.axvline(t_star, color="#c0392b", ls="--", label=f"t*={t_star:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost (INR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_sensitivity_plot(
    curves: dict[str, pd.DataFrame],
    t_stars: dict[str, float],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    colors = {"0.5x FP": "#27ae60", "1.0x FP": "#1f4e79", "2.0x FP": "#c0392b"}
    for name, curve in curves.items():
        ax.plot(curve["threshold"], curve["cost"], lw=2, color=colors.get(name, "gray"), label=name)
        ax.axvline(t_stars[name], color=colors.get(name, "gray"), ls=":", alpha=0.8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost (INR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def eval_a_cost_vector(n: int) -> np.ndarray:
    return np.full(n, EVAL_A_C_FP, dtype=float)


def summarize_eval(
    name: str,
    y: np.ndarray,
    p: np.ndarray,
    t: float,
    scenario: bool,
) -> dict[str, Any]:
    stats = classification_at_threshold(y, p, t)
    payload = {
        "precision": stats["precision"],
        "recall": stats["recall"],
        "f1": stats["f1"],
        "threshold": stats["threshold"],
        "confusion": {k: stats[k] for k in ("tp", "fp", "tn", "fn")},
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "ece": expected_calibration_error(y, p),
        "prevalence": stats["prevalence"],
        "n_test": stats["n"],
    }
    if scenario:
        payload["tag"] = "[SCENARIO]"
        payload["warning"] = "Eval B labels are simulated India/COD RTO, not observed UCI events."
    return payload
