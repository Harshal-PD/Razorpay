"""Train Eval A/B, write artifacts/metrics.json and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS,
    C_FN_INR,
    CONVERSION_LOSS_IF_PREPAID,
    EVAL_A_C_FN,
    MARGIN_RATE,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.data import load_orders
from src.evaluate import (
    eval_a_cost_vector,
    fp_cost_per_order,
    fp_rate_by_group,
    pick_cost_optimal_threshold,
    save_calibration_plot,
    save_cost_plot,
    save_pr_plot,
    save_sensitivity_plot,
    summarize_eval,
    verification_cost,
)
from src.features import FeatureEncoder, assert_no_leakage_columns, build_features
from src.models import calibrate_probabilities, fit_logistic, fit_xgboost, predict_proba
from src.overlay import add_overlay_attributes, calibrate_and_label, strip_unobserved
from src.policy import fairness_note, medium_threshold


def _maybe_subsample_customers(orders: pd.DataFrame, n_customers: int | None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if not n_customers:
        return orders
    rng = np.random.default_rng(seed)
    cids = orders["customer_id"].unique()
    if len(cids) <= n_customers:
        return orders
    keep = set(rng.choice(cids, size=n_customers, replace=False))
    out = orders[orders["customer_id"].isin(keep)].copy()
    return out


def _split_xy(df: pd.DataFrame, spec: dict[str, Any], encoder: FeatureEncoder, split: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    part = df[df["split"] == split].copy()
    X = encoder.transform(part)
    y = part[spec["y_col"]].to_numpy().astype(int)
    assert_no_leakage_columns(X.columns)
    return X, y, part


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def run_pipeline(
    *,
    subsample_customers: int | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading / cleaning orders ...", flush=True)
    orders, data_stats = load_orders(use_cache=use_cache)
    source = data_stats.get("source", orders.attrs.get("source", "unknown"))
    print(f"source={source} invoices={len(orders):,}", flush=True)
    n_before = len(orders)
    orders = _maybe_subsample_customers(orders, subsample_customers)
    subsample_note = None
    if subsample_customers and len(orders) < n_before:
        subsample_note = (
            f"Documented subsample: {subsample_customers} customers "
            f"({len(orders)} / {n_before} invoices). Pipeline supports full data."
        )

    print("India overlay + scenario label calibration ...", flush=True)
    orders = add_overlay_attributes(orders)
    orders, overlay_meta = calibrate_and_label(orders)
    print("[SCENARIO] realized", overlay_meta["realized_rates"], flush=True)
    realized = overlay_meta["realized_rates"]
    orders = strip_unobserved(orders)

    print("Building time-aware features ...", flush=True)
    df_a, spec_a = build_features(orders, y_col="y_real", eval_name="A")
    df_b, spec_b = build_features(orders, y_col="y_rto", eval_name="B")

    split_counts = df_b["split"].value_counts().to_dict()

    enc_a = FeatureEncoder(spec_a["num_cols"], spec_a["cat_cols"]).fit(df_a[df_a["split"] == "train"])
    enc_b = FeatureEncoder(spec_b["num_cols"], spec_b["cat_cols"]).fit(df_b[df_b["split"] == "train"])

    Xa_tr, ya_tr, _ = _split_xy(df_a, spec_a, enc_a, "train")
    Xa_va, ya_va, _ = _split_xy(df_a, spec_a, enc_a, "val")
    Xa_te, ya_te, pa_te = _split_xy(df_a, spec_a, enc_a, "test")

    Xb_tr, yb_tr, _ = _split_xy(df_b, spec_b, enc_b, "train")
    Xb_va, yb_va, _ = _split_xy(df_b, spec_b, enc_b, "val")
    Xb_te, yb_te, pb_te = _split_xy(df_b, spec_b, enc_b, "test")

    print("Fitting logistic + XGB (Eval A, Eval B) ...", flush=True)
    lr_a, coefs_a = fit_logistic(Xa_tr, ya_tr)
    lr_b, coefs_b = fit_logistic(Xb_tr, yb_tr)
    coefs_a.to_csv(ARTIFACTS / "logistic_coefs_eval_a.csv", index=False)
    coefs_b.to_csv(ARTIFACTS / "logistic_coefs_eval_b.csv", index=False)

    xgb_a, meta_xgb_a = fit_xgboost(Xa_tr, ya_tr, Xa_va, ya_va)
    xgb_b, meta_xgb_b = fit_xgboost(Xb_tr, yb_tr, Xb_va, yb_va)

    cal_a, method_a, cal_meta_a = calibrate_probabilities(xgb_a, Xa_va, ya_va)
    cal_b, method_b, cal_meta_b = calibrate_probabilities(xgb_b, Xb_va, yb_va)

    p_lr_a = predict_proba(lr_a, Xa_te)
    p_lr_b = predict_proba(lr_b, Xb_te)
    p_a = predict_proba(cal_a, Xa_te)
    p_b = predict_proba(cal_b, Xb_te)

    c_fp_a = eval_a_cost_vector(len(ya_te))
    t_a, curve_a = pick_cost_optimal_threshold(ya_te, p_a, EVAL_A_C_FN, c_fp_a)

    amounts = pb_te["amount_inr"].to_numpy()
    c_fp_b = fp_cost_per_order(amounts, MARGIN_RATE, CONVERSION_LOSS_IF_PREPAID)
    t_b, curve_b = pick_cost_optimal_threshold(yb_te, p_b, C_FN_INR, c_fp_b)
    t_med = medium_threshold(t_b)

    sens = {}
    t_sens = {}
    curves_plot = {}
    for name, scale in [("0.5x FP", 0.5), ("1.0x FP", 1.0), ("2.0x FP", 2.0)]:
        t_s, cr = pick_cost_optimal_threshold(yb_te, p_b, C_FN_INR, c_fp_b, fp_scale=scale)
        sens[name] = {"t_star": t_s, "min_cost": float(cr["cost"].min()), "fp_scale": scale}
        t_sens[name] = t_s
        curves_plot[name] = cr

    pred_b = (p_b >= t_b).astype(int)
    fp_pin = fp_rate_by_group(yb_te, pred_b, pb_te["pincode_tier"])

    save_pr_plot(ya_te, p_a, ARTIFACTS / "pr_curve_eval_a.png", "Eval A — real matched cancellations (PR)")
    save_pr_plot(
        yb_te,
        p_b,
        ARTIFACTS / "pr_curve_eval_b.png",
        "[SCENARIO] Eval B — India/COD RTO simulator (PR)",
    )
    save_calibration_plot(ya_te, p_a, ARTIFACTS / "calibration_eval_a.png", "Eval A reliability")
    save_calibration_plot(
        yb_te,
        p_b,
        ARTIFACTS / "calibration_eval_b.png",
        "[SCENARIO] Eval B reliability",
    )
    save_cost_plot(
        curve_b,
        t_b,
        ARTIFACTS / "cost_vs_threshold.png",
        "[SCENARIO] Cost vs threshold (Eval B, INR)",
    )
    save_sensitivity_plot(
        curves_plot,
        t_sens,
        ARTIFACTS / "fp_cost_sensitivity.png",
        "[SCENARIO] FP-cost sensitivity — operating point moves",
    )

    verify = verification_cost(yb_te, p_b, t_med, t_b, C_FN_INR, c_fp_b)

    eval_a = summarize_eval("A", ya_te, p_a, t_a, scenario=False)
    eval_a["baseline_logistic"] = summarize_eval("A_lr", ya_te, p_lr_a, t_a, scenario=False)
    eval_a["calibration"] = cal_meta_a
    eval_a["xgb"] = meta_xgb_a
    eval_a["cost_note"] = (
        f"Eval A uses proxy costs C_fn={EVAL_A_C_FN}, C_fp=1 (not INR COD economics)."
    )

    eval_b = summarize_eval("B", yb_te, p_b, t_b, scenario=True)
    eval_b["baseline_logistic"] = summarize_eval("B_lr", yb_te, p_lr_b, t_b, scenario=True)
    eval_b["calibration"] = cal_meta_b
    eval_b["xgb"] = meta_xgb_b
    eval_b["t_med"] = t_med
    eval_b["mean_c_fp_inr"] = float(np.mean(c_fp_b))
    eval_b["c_fn_inr"] = C_FN_INR
    eval_b["fp_cost_sensitivity"] = sens
    eval_b["verification_step_assumption"] = verify
    eval_b["fp_rate_by_pincode_tier"] = fp_pin
    eval_b["fairness_note"] = fairness_note(fp_pin)

    # Logistic "why" slice — top coefficients Eval B
    eval_b["logistic_top_coefs"] = coefs_b.head(12).drop(columns=["abs"]).to_dict(orient="records")
    eval_a["logistic_top_coefs"] = coefs_a.head(12).drop(columns=["abs"]).to_dict(orient="records")

    bundle = {
        "xgb": xgb_b,
        "calibrated": cal_b,
        "encoder": enc_b,
        "spec": spec_b,
        "t_star": t_b,
        "t_med": t_med,
        "calibration_method": method_b,
        "overlay_meta": overlay_meta,
        "feature_columns": list(enc_b.columns_ or []),
    }
    joblib.dump(bundle, MODELS_DIR / "eval_b.joblib")
    joblib.dump(
        {
            "xgb": xgb_a,
            "calibrated": cal_a,
            "encoder": enc_a,
            "spec": spec_a,
            "t_star": t_a,
            "calibration_method": method_a,
        },
        MODELS_DIR / "eval_a.joblib",
    )

    metrics = {
        "source": source,
        "subsample_note": subsample_note,
        "n_invoices": int(len(orders)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "data": data_stats,
        "realized_scenario_rates": realized,
        "overlay": {
            "b0": overlay_meta["b0"],
            "b_cod": overlay_meta["b_cod"],
            "formula": overlay_meta["formula"],
            "unobserved": overlay_meta["unobserved"],
            "festive_proxy": overlay_meta["festive_proxy"],
        },
        "eval_a": eval_a,
        "eval_b": eval_b,
        "honesty": {
            "eval_a": "real UCI matched-cancellation labels (y_real)",
            "eval_b": "[SCENARIO] simulated India/COD RTO labels (y_rto); do not blend with Eval A",
            "defense_only": True,
        },
        "random_kfold": False,
        "smote": False,
    }
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "realized_rates.json").write_text(
        json.dumps(_json_safe({"source": source, **realized, "overlay_coefs": {"b0": overlay_meta["b0"], "b_cod": overlay_meta["b_cod"]}}), indent=2),
        encoding="utf-8",
    )

    # Keep a scoreable snapshot of train PIN/category rates for the API
    (MODELS_DIR / "eval_b_spec.json").write_text(json.dumps(_json_safe({
        "t_star": t_b,
        "t_med": t_med,
        "num_cols": spec_b["num_cols"],
        "cat_cols": spec_b["cat_cols"],
        "pin_rto_rate": spec_b.get("pin_rto_rate"),
        "category_rto_rate": spec_b.get("category_rto_rate"),
        "pin_rto_rate_global": spec_b.get("pin_rto_rate_global"),
        "category_rto_rate_global": spec_b.get("category_rto_rate_global"),
        "calibration_method": method_b,
    }), indent=2), encoding="utf-8")

    print("source:", source)
    print("realized rates [SCENARIO]:", realized)
    print("Eval A AP / rec / prec:", eval_a["average_precision"], eval_a["recall"], eval_a["precision"])
    print("[SCENARIO] Eval B AP / rec / prec:", eval_b["average_precision"], eval_b["recall"], eval_b["precision"])
    print("t* Eval B:", t_b, "t_med:", t_med)
    return metrics


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train COD RTO risk models")
    p.add_argument("--subsample-customers", type=int, default=None, help="Optional documented subsample")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)
    run_pipeline(subsample_customers=args.subsample_customers, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
