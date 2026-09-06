"""Time-aware, leakage-proof order features and temporal split.

PIN/category rates are fit on train orders only, then mapped to val/test.
Customer aggregates exclude the current invoice. No cancel dates in features.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import BURNIN_END, FASHION_KEYWORDS, TRAIN_END, VAL_END


EVAL_A_NUM = [
    "amount_gbp",
    "log_amount_gbp",
    "qty",
    "n_lines",
    "account_age_days",
    "prior_order_count",
    "prior_return_count",
    "prior_return_rate",
    "first_time_flag",
    "is_fashion_kw",
    "fashion_return_rate",
    "dow",
    "is_festive",
    "is_uk",
]
EVAL_A_CAT = ["price_tier"]

EVAL_B_NUM = [
    "amount_inr",
    "log_amount_inr",
    "qty",
    "n_lines",
    "is_cod",
    "account_age_days",
    "prior_order_count",
    "prior_return_count",
    "prior_return_rate",
    "first_time_flag",
    "first_time_cod_flag",
    "category_rto_rate",
    "is_fashion",
    "pin_rto_rate",
    "cod_x_pin_rto",
    "cod_x_fashion",
    "cod_x_festive",
    "cod_x_prior_return",
    "days_to_expected_delivery",
    "dow",
    "is_festive",
]
EVAL_B_CAT = ["price_tier", "product_category", "pincode_tier", "delivery_zone"]

FORBIDDEN_FEATURE_COLS = {
    "z_customer",
    "eps",
    "y_real",
    "y_rto",
    "matched_credited_value",
    "return_value_frac",
    "cancel_date",
    "InvoiceDate_cancel",
}


def assign_split(invoice_date: pd.Series) -> pd.Series:
    dt = pd.to_datetime(invoice_date)
    split = pd.Series("test", index=dt.index)
    split = split.mask(dt < pd.Timestamp(VAL_END), "val")
    split = split.mask(dt < pd.Timestamp(TRAIN_END), "train")
    split = split.mask(dt < pd.Timestamp(BURNIN_END), "burnin")
    return split.rename("split")


def keyword_fashion(description: pd.Series) -> pd.Series:
    text = description.fillna("").astype(str).str.lower()
    pat = "|".join(FASHION_KEYWORDS)
    return text.str.contains(pat, regex=True).astype(int).rename("is_fashion_kw")


def add_customer_history(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Lagged customer stats as-of invoice_date. Current invoice excluded from aggregates."""
    out = df.sort_values(["customer_id", "invoice_date", "invoice"]).copy()
    g = out.groupby("customer_id", sort=False)
    out["prior_order_count"] = g.cumcount()
    out["prior_return_count"] = g[y_col].cumsum() - out[y_col]
    out["prior_return_rate"] = np.where(
        out["prior_order_count"] > 0,
        out["prior_return_count"] / out["prior_order_count"],
        0.0,
    )
    out["first_time_flag"] = (out["prior_order_count"] == 0).astype(int)
    first_seen = g["invoice_date"].transform("min")
    out["account_age_days"] = (pd.to_datetime(out["invoice_date"]) - pd.to_datetime(first_seen)).dt.days
    if "is_cod" in out.columns:
        out["first_time_cod_flag"] = (out["first_time_flag"] * out["is_cod"]).astype(int)
    else:
        out["first_time_cod_flag"] = 0
    return out


def fit_group_rate(
    train: pd.DataFrame,
    col: str,
    y_col: str,
    min_n: int = 25,
) -> tuple[dict[Any, float], float]:
    global_rate = float(train[y_col].mean())
    stats = train.groupby(col)[y_col].agg(["mean", "size"])
    rates = {}
    for key, row in stats.iterrows():
        rates[key] = float(row["mean"]) if row["size"] >= min_n else global_rate
    return rates, global_rate


def map_group_rate(series: pd.Series, mapping: dict[Any, float], global_rate: float) -> pd.Series:
    return series.map(mapping).fillna(global_rate).astype(float)


def fit_price_bins(train_amount: pd.Series, q: int = 3) -> np.ndarray:
    _, bins = pd.qcut(train_amount, q=q, retbins=True, duplicates="drop")
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def apply_price_tier(amount: pd.Series, bins: np.ndarray) -> pd.Series:
    labels = ["low", "mid", "high"][: len(bins) - 1]
    return pd.cut(amount, bins=bins, labels=labels, include_lowest=True).astype(str)


def build_features(
    orders: pd.DataFrame,
    y_col: str,
    eval_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute time-aware features. Train-only PIN/category/fashion rates."""
    df = orders.copy()
    df["split"] = assign_split(df["invoice_date"])
    df["dow"] = pd.to_datetime(df["invoice_date"]).dt.dayofweek
    if "is_festive" not in df.columns:
        df["is_festive"] = pd.to_datetime(df["invoice_date"]).dt.month.isin([11, 12]).astype(int)
    df["is_uk"] = (df["country"] == "United Kingdom").astype(int)
    df["is_fashion_kw"] = keyword_fashion(df["description"] if "description" in df.columns else pd.Series("", index=df.index))

    df = add_customer_history(df, y_col=y_col)
    train = df[df["split"] == "train"]
    if train.empty:
        raise ValueError("No train rows after temporal split — check invoice dates.")

    amount_col = "amount_inr" if "amount_inr" in df.columns and eval_name == "B" else "amount_gbp"
    bins = fit_price_bins(train[amount_col])
    df["price_tier"] = apply_price_tier(df[amount_col], bins)

    spec: dict[str, Any] = {"y_col": y_col, "eval": eval_name, "price_bins": bins.tolist()}

    if eval_name == "A":
        df["log_amount_gbp"] = np.log1p(df["amount_gbp"].clip(lower=0))
        fashion_rates, g = fit_group_rate(train, "is_fashion_kw", y_col)
        df["fashion_return_rate"] = map_group_rate(df["is_fashion_kw"], fashion_rates, g)
        spec["fashion_return_rate"] = {str(k): v for k, v in fashion_rates.items()}
        spec["fashion_return_rate_global"] = g
        num_cols, cat_cols = list(EVAL_A_NUM), list(EVAL_A_CAT)
    else:
        cat_rates, gcat = fit_group_rate(train, "product_category", y_col)
        pin_rates, gpin = fit_group_rate(train, "pincode_tier", y_col)
        df["category_rto_rate"] = map_group_rate(df["product_category"], cat_rates, gcat)
        df["pin_rto_rate"] = map_group_rate(df["pincode_tier"], pin_rates, gpin)
        # Honest interactions from already-observable fields (no z_customer/eps).
        # A linear model cannot discover these on its own; they sharpen the baseline
        # and give XGB pre-formed COD-conditioned signal.
        df["log_amount_inr"] = np.log1p(df["amount_inr"].clip(lower=0))
        df["cod_x_pin_rto"] = df["is_cod"].astype(float) * df["pin_rto_rate"].astype(float)
        df["cod_x_fashion"] = df["is_cod"].astype(float) * df["is_fashion"].astype(float)
        df["cod_x_festive"] = df["is_cod"].astype(float) * df["is_festive"].astype(float)
        df["cod_x_prior_return"] = df["is_cod"].astype(float) * df["prior_return_rate"].astype(float)
        spec["category_rto_rate"] = dict(cat_rates)
        spec["category_rto_rate_global"] = gcat
        spec["pin_rto_rate"] = dict(pin_rates)
        spec["pin_rto_rate_global"] = gpin
        num_cols, cat_cols = list(EVAL_B_NUM), list(EVAL_B_CAT)

    spec["num_cols"] = num_cols
    spec["cat_cols"] = cat_cols
    feat_cols = num_cols + cat_cols
    leaked = FORBIDDEN_FEATURE_COLS.intersection(feat_cols)
    if leaked:
        raise RuntimeError(f"Forbidden columns in feature set: {leaked}")
    return df, spec


def modeling_frame(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    cols = spec["num_cols"] + spec["cat_cols"] + [spec["y_col"], "split", "invoice", "customer_id"]
    extra = [c for c in ("amount_inr", "pincode_tier", "is_cod") if c in df.columns and c not in cols]
    return df[cols + extra].copy()


class FeatureEncoder:
    """Train-fit numeric fill + one-hot; transform val/test with aligned columns."""

    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_median_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self.num_median_ = df[self.num_cols].median(numeric_only=True)
        X = self._matrix(df)
        self.columns_ = list(X.columns)
        return self

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.num_median_ is not None
        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.num_median_)
        cat = pd.get_dummies(df[self.cat_cols].astype(str), prefix=self.cat_cols, dtype=int)
        return pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("FeatureEncoder not fit")
        X = self._matrix(df)
        return X.reindex(columns=self.columns_, fill_value=0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def assert_no_leakage_columns(columns: Iterable[str]) -> None:
    bad = FORBIDDEN_FEATURE_COLS.intersection(set(columns))
    if bad:
        raise AssertionError(f"Leakage / unobserved columns present: {bad}")


def inference_row(payload: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row Eval B frame from an API/demo payload using train-only mappings."""
    from datetime import datetime as _dt

    tier = str(payload.get("pincode_tier") or "tier2")
    cat = str(payload.get("product_category") or "fashion")
    zone = {"metro": "local", "tier2": "regional", "tier3": "remote"}.get(tier, "regional")
    days = {"local": 2, "regional": 4, "remote": 7}[zone]
    raw_date = payload.get("invoice_date")
    if raw_date:
        dt = _dt.fromisoformat(str(raw_date)[:19])
    else:
        dt = _dt(2011, 11, 15)
    prior_n = int(payload.get("prior_order_count") or 0)
    prior_ret = float(payload.get("prior_return_rate") or 0.0)
    prior_c = int(payload.get("prior_return_count") or round(prior_ret * prior_n))
    first = payload.get("first_time")
    if first is None:
        first = prior_n == 0
    first = int(bool(first))
    is_cod = int(bool(payload.get("is_cod", True)))
    amount = float(payload.get("amount_inr") or 0)
    bins = spec.get("price_bins")
    if bins:
        price_tier = apply_price_tier(pd.Series([amount]), np.array(bins, dtype=float)).iloc[0]
    else:
        price_tier = "mid"
    pin_map = spec.get("pin_rto_rate") or {}
    cat_map = spec.get("category_rto_rate") or {}
    gpin = float(spec.get("pin_rto_rate_global") or 0.2)
    gcat = float(spec.get("category_rto_rate_global") or 0.2)
    pin_rto = float(pin_map.get(tier, gpin))
    is_fashion = int(cat == "fashion")
    is_festive = int(dt.month in (11, 12) or bool(payload.get("is_festive")))
    row = {
        "amount_inr": amount,
        "log_amount_inr": float(np.log1p(max(amount, 0.0))),
        "qty": int(payload.get("qty") or 1),
        "n_lines": int(payload.get("n_lines") or 1),
        "is_cod": is_cod,
        "account_age_days": int(payload.get("account_age_days") or 0),
        "prior_order_count": prior_n,
        "prior_return_count": prior_c,
        "prior_return_rate": prior_ret,
        "first_time_flag": first,
        "first_time_cod_flag": int(first and is_cod),
        "category_rto_rate": float(cat_map.get(cat, gcat)),
        "is_fashion": is_fashion,
        "pin_rto_rate": pin_rto,
        "cod_x_pin_rto": float(is_cod * pin_rto),
        "cod_x_fashion": float(is_cod * is_fashion),
        "cod_x_festive": float(is_cod * is_festive),
        "cod_x_prior_return": float(is_cod * prior_ret),
        "days_to_expected_delivery": int(payload.get("days_to_expected_delivery") or days),
        "dow": int(dt.weekday()),
        "is_festive": is_festive,
        "price_tier": str(price_tier),
        "product_category": cat,
        "pincode_tier": tier,
        "delivery_zone": zone,
    }
    return pd.DataFrame([row])
