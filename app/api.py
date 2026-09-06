"""FastAPI scorer: POST /v1/risk/rto — defense-only, never silent deny."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.features import inference_row
from src.policy import decide

app = FastAPI(
    title="COD RTO Risk Scorer",
    version="0.1.0",
    description="Track 02 defense-only RTO risk at order confirmation. Never silent deny.",
)

_BUNDLE: dict[str, Any] | None = None
_EXPLAINER = None


class RTORequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, description="Omit to trigger insufficient_history")
    amount_inr: Optional[float] = None
    is_cod: bool = True
    product_category: str = "fashion"
    pincode_tier: str = "tier2"
    qty: int = 1
    n_lines: int = 1
    prior_order_count: int = 0
    prior_return_rate: float = 0.0
    prior_return_count: Optional[int] = None
    first_time: Optional[bool] = None
    account_age_days: int = 0
    invoice_date: Optional[str] = Field(default="2011-11-15", description="ISO date; Nov–Dec is festive proxy")
    is_festive: Optional[bool] = None


class RTOResponse(BaseModel):
    rto_probability: Optional[float]
    risk_tier: str
    recommended_action: str
    top_factors: list[dict[str, Any]]
    policy: str
    reason: str
    eval: str = "B"
    tag: str = "[SCENARIO]"


def load_bundle() -> dict[str, Any] | None:
    global _BUNDLE
    path = MODELS_DIR / "eval_b.joblib"
    if _BUNDLE is None and path.exists():
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def _explainer(xgb):
    global _EXPLAINER
    if _EXPLAINER is None:
        import shap

        _EXPLAINER = shap.TreeExplainer(xgb)
    return _EXPLAINER


def _top_factors(xgb, X, k: int = 3) -> list[dict[str, Any]]:
    try:
        import xgboost as xgb_lib

        names = list(X.columns)
        dmat = xgb_lib.DMatrix(X.to_numpy(), feature_names=names)
        contribs = xgb.get_booster().predict(dmat, pred_contribs=True)
        row = np.asarray(contribs)[0, :-1]  # drop bias
        order = np.argsort(-np.abs(row))[:k]
        out = []
        for i in order:
            val = X.iloc[0, i]
            try:
                val_out = round(float(val), 4)
            except (TypeError, ValueError):
                val_out = str(val)
            out.append({"feature": names[i], "shap": round(float(row[i]), 4), "value": val_out})
        return out
    except Exception:
        try:
            import shap

            exp = _explainer(xgb)
            vals = exp.shap_values(X.to_numpy())
            if isinstance(vals, list):
                vals = vals[1]
            row = np.asarray(vals)[0]
            names = list(X.columns)
            order = np.argsort(-np.abs(row))[:k]
            out = []
            for i in order:
                out.append(
                    {
                        "feature": names[i],
                        "shap": round(float(row[i]), 4),
                        "value": round(float(X.iloc[0, i]), 4)
                        if np.isfinite(X.iloc[0, i])
                        else str(X.iloc[0, i]),
                    }
                )
            return out
        except Exception:
            return []


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_bundle()
    return {"ok": True, "model_loaded": bundle is not None, "policy": "never_silent_deny"}


@app.post("/v1/risk/rto", response_model=RTOResponse)
def score_rto(req: RTORequest) -> RTOResponse:
    bundle = load_bundle()
    features_ok = (
        req.customer_id is not None
        and str(req.customer_id).strip() != ""
        and req.amount_inr is not None
        and req.amount_inr > 0
        and req.pincode_tier in {"metro", "tier2", "tier3"}
        and req.product_category in {"fashion", "electronics", "beauty", "home"}
    )

    if not features_ok or bundle is None:
        payload = decide(None, t_med=0.18, t_star=0.42, features_ok=False)
        payload["top_factors"] = []
        if bundle is None:
            payload["reason"] = "insufficient_history"
        return RTOResponse(**payload)

    spec = bundle["spec"]
    row = inference_row(req.model_dump(), spec)
    X = bundle["encoder"].transform(row)
    proba = float(bundle["calibrated"].predict_proba(X)[:, 1][0])
    factors = _top_factors(bundle["xgb"], X, k=3)
    t_star = float(bundle["t_star"])
    t_med = float(bundle["t_med"])
    payload = decide(proba, t_med=t_med, t_star=t_star, features_ok=True, top_factors=factors)
    return RTOResponse(**payload)
