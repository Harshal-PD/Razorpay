"""Leakage tests: no cancel dates, current invoice excluded, train-only PIN rates."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FORBIDDEN_FEATURE_COLS,
    FeatureEncoder,
    add_customer_history,
    assign_split,
    build_features,
    fit_group_rate,
    map_group_rate,
)
from src.overlay import add_overlay_attributes, calibrate_and_label, strip_unobserved


def _tiny_orders() -> pd.DataFrame:
    """Hand-built invoices spanning burn-in / train / val / test."""
    rows = [
        # customer 1: burn-in then train then test
        {"invoice": "A1", "invoice_date": datetime(2010, 1, 10), "customer_id": 1, "country": "United Kingdom",
         "qty": 2, "n_lines": 1, "amount_gbp": 20.0, "description": "JUMBO BAG RED", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        {"invoice": "A2", "invoice_date": datetime(2010, 6, 10), "customer_id": 1, "country": "United Kingdom",
         "qty": 3, "n_lines": 2, "amount_gbp": 40.0, "description": "CAKE STAND", "y_real": 1, "matched_credited_value": 40.0, "return_value_frac": 1.0, "is_partial_return": 0},
        {"invoice": "A3", "invoice_date": datetime(2011, 11, 20), "customer_id": 1, "country": "United Kingdom",
         "qty": 1, "n_lines": 1, "amount_gbp": 15.0, "description": "PARTY BUNTING", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        # customer 2: train only, two orders
        {"invoice": "B1", "invoice_date": datetime(2010, 4, 1), "customer_id": 2, "country": "United Kingdom",
         "qty": 5, "n_lines": 2, "amount_gbp": 80.0, "description": "NECKLACE", "y_real": 1, "matched_credited_value": 80.0, "return_value_frac": 1.0, "is_partial_return": 0},
        {"invoice": "B2", "invoice_date": datetime(2011, 1, 15), "customer_id": 2, "country": "United Kingdom",
         "qty": 2, "n_lines": 1, "amount_gbp": 25.0, "description": "MUG", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        # fill train/val/test with more customers so PIN rates have counts
        {"invoice": "C1", "invoice_date": datetime(2010, 5, 1), "customer_id": 3, "country": "France",
         "qty": 1, "n_lines": 1, "amount_gbp": 12.0, "description": "POSTER", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        {"invoice": "C2", "invoice_date": datetime(2011, 9, 12), "customer_id": 3, "country": "France",
         "qty": 2, "n_lines": 1, "amount_gbp": 18.0, "description": "POSTER", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        {"invoice": "D1", "invoice_date": datetime(2011, 10, 2), "customer_id": 4, "country": "United Kingdom",
         "qty": 4, "n_lines": 2, "amount_gbp": 30.0, "description": "BAG", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
        {"invoice": "E1", "invoice_date": datetime(2011, 12, 1), "customer_id": 5, "country": "United Kingdom",
         "qty": 8, "n_lines": 3, "amount_gbp": 90.0, "description": "DRESS", "y_real": 0, "matched_credited_value": 0.0, "return_value_frac": 0.0, "is_partial_return": 0},
    ]
    # extra train rows for qcut / group rates
    for i in range(6, 40):
        rows.append(
            {
                "invoice": f"T{i}",
                "invoice_date": datetime(2010, 8, 1 + (i % 27)),
                "customer_id": 100 + i,
                "country": "United Kingdom",
                "qty": 2,
                "n_lines": 1,
                "amount_gbp": 10.0 + i,
                "description": "HOLDER",
                "y_real": int(i % 7 == 0),
                "matched_credited_value": 0.0,
                "return_value_frac": 0.0,
                "is_partial_return": 0,
            }
        )
    return pd.DataFrame(rows)


def test_no_cancel_date_in_features():
    orders = _tiny_orders()
    orders = add_overlay_attributes(orders)
    orders, _ = calibrate_and_label(orders)
    orders = strip_unobserved(orders)
    df, spec = build_features(orders, y_col="y_rto", eval_name="B")
    feat = spec["num_cols"] + spec["cat_cols"]
    assert "cancel_date" not in feat
    assert "InvoiceDate_cancel" not in df.columns
    leaked = FORBIDDEN_FEATURE_COLS.intersection(feat)
    assert not leaked
    enc = FeatureEncoder(spec["num_cols"], spec["cat_cols"]).fit(df[df["split"] == "train"])
    X = enc.transform(df[df["split"] == "test"])
    assert not FORBIDDEN_FEATURE_COLS.intersection(X.columns)
    assert "z_customer" not in X.columns
    assert "eps" not in X.columns
    assert "y_rto" not in X.columns


def test_customer_aggregates_exclude_current_invoice():
    df = _tiny_orders()
    hist = add_customer_history(df, y_col="y_real")
    b1 = hist.loc[hist["invoice"] == "B1"].iloc[0]
    b2 = hist.loc[hist["invoice"] == "B2"].iloc[0]
    assert b1["prior_order_count"] == 0
    assert b1["prior_return_count"] == 0
    assert b1["first_time_flag"] == 1
    # Current y_real=1 on B1 must not appear in B1's own prior_* 
    assert b2["prior_order_count"] == 1
    assert b2["prior_return_count"] == 1
    assert b2["prior_return_rate"] == pytest.approx(1.0)
    # B2's own y=0 is not counted in its prior
    assert b2["y_real"] == 0


def test_pin_rates_are_train_mapping_not_refit_on_test():
    """PIN historical RTO on test equals the train mapping, even if test labels differ."""
    rng = np.random.default_rng(0)
    n_train, n_test = 80, 40
    rows = []
    for i in range(n_train):
        tier = "metro" if i < 40 else "tier3"
        rows.append(
            {
                "invoice": f"tr{i}",
                "invoice_date": datetime(2010, 6, 1),
                "customer_id": i,
                "country": "United Kingdom",
                "qty": 1,
                "n_lines": 1,
                "amount_gbp": 20.0,
                "description": "X",
                "y_real": 0,
                "matched_credited_value": 0.0,
                "return_value_frac": 0.0,
                "is_partial_return": 0,
                "product_category": "home",
                "pincode_tier": tier,
                "is_cod": 1,
                "is_fashion": 0,
                "is_festive": 0,
                "delivery_zone": "local" if tier == "metro" else "remote",
                "days_to_expected_delivery": 2,
                "amount_inr": 800.0,
                "y_rto": 0 if tier == "metro" else 1,  # train: metro 0%, tier3 100%
            }
        )
    for i in range(n_test):
        tier = "metro" if i < 20 else "tier3"
        rows.append(
            {
                "invoice": f"te{i}",
                "invoice_date": datetime(2011, 11, 15),
                "customer_id": 1000 + i,
                "country": "United Kingdom",
                "qty": 1,
                "n_lines": 1,
                "amount_gbp": 20.0,
                "description": "X",
                "y_real": 0,
                "matched_credited_value": 0.0,
                "return_value_frac": 0.0,
                "is_partial_return": 0,
                "product_category": "home",
                "pincode_tier": tier,
                "is_cod": 1,
                "is_fashion": 0,
                "is_festive": 1,
                "delivery_zone": "local" if tier == "metro" else "remote",
                "days_to_expected_delivery": 2,
                "amount_inr": 800.0,
                # test labels inverted so a leaky re-fit would flip the rates
                "y_rto": 1 if tier == "metro" else 0,
            }
        )
    df0 = pd.DataFrame(rows)
    df, spec = build_features(df0, y_col="y_rto", eval_name="B")
    train_map = spec["pin_rto_rate"]
    g = spec["pin_rto_rate_global"]
    test = df[df["split"] == "test"]
    mapped = map_group_rate(test["pincode_tier"], train_map, g)
    assert np.allclose(test["pin_rto_rate"].to_numpy(), mapped.to_numpy())
    # Train metro rate is ~0, must not become 1 just because test metro is all-positive
    assert train_map["metro"] == pytest.approx(0.0)
    assert train_map["tier3"] == pytest.approx(1.0)
    metro_test_rate = test.loc[test["pincode_tier"] == "metro", "pin_rto_rate"].iloc[0]
    assert metro_test_rate == pytest.approx(train_map["metro"])
    assert metro_test_rate != pytest.approx(test.loc[test["pincode_tier"] == "metro", "y_rto"].mean())


def test_temporal_split_not_random_kfold():
    dates = pd.Series(
        [
            datetime(2009, 12, 15),
            datetime(2010, 6, 1),
            datetime(2011, 9, 15),
            datetime(2011, 11, 20),
        ]
    )
    s = assign_split(dates)
    assert list(s) == ["burnin", "train", "val", "test"]


def test_unmatched_cancels_are_not_phantom_orders():
    from src.data import build_invoice_table, match_cancels_to_sales

    sales = pd.DataFrame(
        {
            "Invoice": ["10001"],
            "StockCode": ["111"],
            "Description": ["MUG"],
            "Quantity": [2],
            "InvoiceDate": [pd.Timestamp("2010-06-01")],
            "Price": [3.5],
            "Customer ID": [1],
            "Country": ["United Kingdom"],
        }
    )
    cancels = pd.DataFrame(
        {
            "Invoice": ["C99999"],
            "StockCode": ["999"],
            "Description": ["GHOST"],
            "Quantity": [-2],
            "InvoiceDate": [pd.Timestamp("2010-06-10")],
            "Price": [3.5],
            "Customer ID": [99],
            "Country": ["United Kingdom"],
        }
    )
    credited, stats = match_cancels_to_sales(sales, cancels)
    orders = build_invoice_table(sales, credited)
    assert "C99999" not in set(orders["invoice"].astype(str))
    assert stats["n_unmatched_cancel_lines"] == 1
    assert orders["y_real"].iloc[0] == 0


def test_fit_group_rate_ignores_heldout():
    train = pd.DataFrame({"pin": ["metro", "metro", "tier3"] * 20, "y": [0, 0, 1] * 20})
    mapping, g = fit_group_rate(train, "pin", "y", min_n=5)
    heldout = pd.Series(["metro", "tier3", "unknown"])
    mapped = map_group_rate(heldout, mapping, g)
    assert mapped.iloc[0] == pytest.approx(mapping["metro"])
    assert mapped.iloc[1] == pytest.approx(mapping["tier3"])
    assert mapped.iloc[2] == pytest.approx(g)
