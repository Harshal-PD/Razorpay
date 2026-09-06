"""India/COD scenario overlay (Eval B only). Synthetic and named as such.

Generator includes unobserved per-customer latent `z_customer` and per-order `eps`
that are NEVER features — this is what keeps AUC honest (~0.70–0.85, not 0.99).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_INR_MEDIAN,
    CATEGORY_PROBS,
    COD_BASE_RATE,
    OVERLAY_SEED,
    PIN_PREFIX_TABLE,
    PIN_PRIOR,
    TARGET_COD,
    TARGET_PREPAID,
)

# Documented generator coefficients (b0, b_cod calibrated empirically)
B_PIN = 0.70
B_FASHION = 0.55
B_FESTIVE = 0.38
B_HIST = 1.25
Z_SCALE = 0.4
EPS_SCALE = 0.8


def _stable_unit(key: str) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def assign_pincode_tier(country: str, customer_id: int) -> str:
    """Country + hash(customer) → metro/tier2/tier3. Not a blocklist."""
    u = _stable_unit(f"tier:{int(customer_id)}")
    if str(country) != "United Kingdom":
        if u < 0.70:
            return "metro"
        if u < 0.92:
            return "tier2"
        return "tier3"
    if u < 0.35:
        return "metro"
    if u < 0.70:
        return "tier2"
    return "tier3"


def assign_pin_prefix(tier: str, customer_id: int) -> str:
    table = PIN_PREFIX_TABLE[tier]
    idx = int(_stable_unit(f"pin:{int(customer_id)}") * len(table)) % len(table)
    return table[idx]


def map_amount_inr(amount_gbp: pd.Series) -> pd.Series:
    """Percentile map onto D2C-like INR AOV. Does NOT FX-convert wholesale GBP."""
    pct = amount_gbp.rank(method="average", pct=True).to_numpy()
    # p0, p10, p50, p90, p100 → log INR (median ~1100)
    xs = np.array([0.0, 0.10, 0.50, 0.90, 1.0])
    ys = np.log(np.array([199.0, 350.0, AMOUNT_INR_MEDIAN, 3500.0, 8999.0]))
    log_amt = np.interp(pct, xs, ys)
    return pd.Series(np.round(np.exp(log_amt), 2), index=amount_gbp.index, name="amount_inr")


def _draw_categories(n: int, rng: np.random.Generator) -> np.ndarray:
    labels = list(CATEGORY_PROBS.keys())
    p = np.array([CATEGORY_PROBS[k] for k in labels], dtype=float)
    p = p / p.sum()
    return rng.choice(labels, size=n, p=p)


def assign_is_cod(
    first_time: pd.Series,
    pincode_tier: pd.Series,
    rng: np.random.Generator,
    base: float = COD_BASE_RATE,
) -> pd.Series:
    """~60% COD, higher for first-time and tier3. Does not read y_real."""
    p = np.full(len(first_time), base, dtype=float)
    p = p + 0.14 * first_time.to_numpy().astype(float)
    tier = pincode_tier.to_numpy()
    p = p + 0.11 * (tier == "tier3").astype(float)
    p = p - 0.08 * (tier == "metro").astype(float)
    p = np.clip(p, 0.22, 0.88)
    draws = rng.random(len(p)) < p
    return pd.Series(draws.astype(int), index=first_time.index, name="is_cod")


def _first_time_flags(orders: pd.DataFrame) -> pd.Series:
    tmp = orders.sort_values(["customer_id", "invoice_date", "invoice"])
    flag = tmp.groupby("customer_id").cumcount().eq(0).astype(int)
    return flag.reindex(orders.index)


def add_overlay_attributes(orders: pd.DataFrame, seed: int = OVERLAY_SEED) -> pd.DataFrame:
    """Attach synthetic India fields. Does not create y_rto yet."""
    rng = np.random.default_rng(seed)
    out = orders.copy()
    out["amount_inr"] = map_amount_inr(out["amount_gbp"])
    out["product_category"] = _draw_categories(len(out), rng)
    out["pincode_tier"] = [
        assign_pincode_tier(c, int(cid)) for c, cid in zip(out["country"], out["customer_id"])
    ]
    out["pin_prefix"] = [
        assign_pin_prefix(t, int(cid)) for t, cid in zip(out["pincode_tier"], out["customer_id"])
    ]
    out["is_fashion"] = (out["product_category"] == "fashion").astype(int)
    month = pd.to_datetime(out["invoice_date"]).dt.month
    out["is_festive"] = month.isin([11, 12]).astype(int)
    zone_map = {"metro": "local", "tier2": "regional", "tier3": "remote"}
    out["delivery_zone"] = out["pincode_tier"].map(zone_map)
    days_map = {"local": 2, "regional": 4, "remote": 7}
    noise = (rng.integers(0, 2, size=len(out))).astype(int)
    out["days_to_expected_delivery"] = out["delivery_zone"].map(days_map).astype(int) + noise
    first_time = _first_time_flags(out)
    out["overlay_first_time"] = first_time
    out["is_cod"] = assign_is_cod(first_time, out["pincode_tier"], rng)
    # Unobserved latents — attached for generation, stripped before modeling
    cust_ids = out["customer_id"].unique()
    z_vals = rng.normal(0.0, 1.0, size=len(cust_ids))
    z_map = dict(zip(cust_ids, z_vals))
    out["z_customer"] = out["customer_id"].map(z_map)
    out["eps"] = rng.normal(0.0, 1.0, size=len(out))
    return out


def _generate_y_rto(orders: pd.DataFrame, b0: float, b_cod: float) -> np.ndarray:
    """Sequential Bernoulli labels using lagged y_rto as past_return_rate."""
    df = orders.sort_values(["invoice_date", "invoice"]).copy()
    y = np.zeros(len(df), dtype=int)
    n_prev: dict[int, int] = {}
    pos_prev: dict[int, int] = {}
    pin_prior = df["pincode_tier"].map(PIN_PRIOR).astype(float).to_numpy()
    is_cod = df["is_cod"].to_numpy()
    is_fashion = df["is_fashion"].to_numpy()
    is_festive = df["is_festive"].to_numpy()
    z = df["z_customer"].to_numpy()
    eps = df["eps"].to_numpy()
    cids = df["customer_id"].to_numpy()
    invoices = df["invoice"].astype(str).to_numpy()

    for i in range(len(df)):
        cid = int(cids[i])
        n = n_prev.get(cid, 0)
        past = (pos_prev.get(cid, 0) / n) if n else 0.0
        logit = (
            b0
            + b_cod * is_cod[i]
            + B_PIN * pin_prior[i]
            + B_FASHION * is_fashion[i]
            + B_FESTIVE * is_festive[i]
            + B_HIST * past
            + Z_SCALE * z[i]
            + EPS_SCALE * eps[i]
        )
        p = float(_sigmoid(logit))
        # Reproducible Uniform(0,1) vs p (independent of z/eps except through p).
        u = _stable_unit(f"y:{invoices[i]}:{i}")
        yi = int(u < p)
        y[i] = yi
        n_prev[cid] = n + 1
        pos_prev[cid] = pos_prev.get(cid, 0) + yi

    # Reindex to original order
    y_series = pd.Series(y, index=df.index)
    return y_series.reindex(orders.index).to_numpy()


def _realized_rates(orders: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    is_cod = orders["is_cod"].to_numpy()
    fashion = orders["is_fashion"].to_numpy()
    tier3 = (orders["pincode_tier"] == "tier3").to_numpy()
    metro = (orders["pincode_tier"] == "metro").to_numpy()
    prepaid_m = is_cod == 0
    return {
        "overall": float(y.mean()),
        "prepaid": float(y[prepaid_m].mean()) if prepaid_m.any() else float("nan"),
        "cod": float(y[is_cod == 1].mean()) if (is_cod == 1).any() else float("nan"),
        "fashion_cod": float(y[(is_cod == 1) & (fashion == 1)].mean())
        if ((is_cod == 1) & (fashion == 1)).any()
        else float("nan"),
        "tier3_cod": float(y[(is_cod == 1) & tier3].mean()) if ((is_cod == 1) & tier3).any() else float("nan"),
        "metro_cod": float(y[(is_cod == 1) & metro].mean()) if ((is_cod == 1) & metro).any() else float("nan"),
        "cod_share": float(is_cod.mean()),
    }


def _score_targets(rates: dict[str, float]) -> float:
    prepaid = rates["prepaid"]
    cod = rates["cod"]
    loss = 2.5 * (prepaid - TARGET_PREPAID) ** 2 + (cod - TARGET_COD) ** 2
    if rates["fashion_cod"] < rates["cod"] - 1e-6:
        loss += 0.05
    if rates["tier3_cod"] <= rates["metro_cod"]:
        loss += 0.04
    if not (0.04 <= prepaid <= 0.09):
        loss += 0.08
    if not (0.26 <= cod <= 0.34):
        loss += 0.08
    return loss


def calibrate_and_label(
    orders: pd.DataFrame,
    seed: int = OVERLAY_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search b0, b_cod so realized prepaid ~5–7% and COD ~28–32%."""
    if "z_customer" not in orders.columns:
        orders = add_overlay_attributes(orders, seed=seed)

    best = None
    # Coarse grid then local refine
    for b0 in np.linspace(-3.6, -2.0, 17):
        for b_cod in np.linspace(1.15, 2.35, 13):
            y = _generate_y_rto(orders, float(b0), float(b_cod))
            rates = _realized_rates(orders, y)
            loss = _score_targets(rates)
            if best is None or loss < best[0]:
                best = (loss, float(b0), float(b_cod), y, rates)

    assert best is not None
    _, b0, b_cod, y, rates = best
    # Local refine
    for db0 in (-0.08, 0.0, 0.08):
        for dbc in (-0.08, 0.0, 0.08):
            y2 = _generate_y_rto(orders, b0 + db0, b_cod + dbc)
            rates2 = _realized_rates(orders, y2)
            loss2 = _score_targets(rates2)
            if loss2 < best[0]:
                best = (loss2, b0 + db0, b_cod + dbc, y2, rates2)
                y, rates, b0, b_cod = y2, rates2, b0 + db0, b_cod + dbc

    out = orders.copy()
    out["y_rto"] = y.astype(int)
    meta = {
        "b0": b0,
        "b_cod": b_cod,
        "b_pin": B_PIN,
        "b_fashion": B_FASHION,
        "b_festive": B_FESTIVE,
        "b_hist": B_HIST,
        "z_scale": Z_SCALE,
        "eps_scale": EPS_SCALE,
        "formula": (
            "logit = b0 + b_cod*COD + b_pin*pin_prior + b_fashion*fashion "
            "+ b_festive*festive + b_hist*past_return_rate + 0.4*z_customer + 0.8*eps"
        ),
        "unobserved": ["z_customer", "eps"],
        "realized_rates": rates,
        "targets": {"prepaid": TARGET_PREPAID, "cod": TARGET_COD},
        "pin_prior": dict(PIN_PRIOR),
        "category_probs": dict(CATEGORY_PROBS),
        "cod_base_rate_target": COD_BASE_RATE,
        "festive_proxy": "UK Nov–Dec invoice dates used as India's festive-spike proxy",
    }
    return out, meta


def strip_unobserved(df: pd.DataFrame) -> pd.DataFrame:
    """Drop latents the model must not see."""
    return df.drop(columns=["z_customer", "eps"], errors="ignore")
