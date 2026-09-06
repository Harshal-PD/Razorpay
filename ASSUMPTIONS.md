# ASSUMPTIONS.md

This document is the honesty contract for the COD RTO Risk Scorer (Razorpay Buildathon Track 02 — AI Risk Manager).

Nothing here is claimed as a causal measurement of Indian COD RTO on UCI data. UCI is a **UK giftware wholesaler, December 2009–December 2011**. Cancellations (`Invoice` prefix `C`, negative quantity) are real, and they are **not** India cash-on-delivery return-to-origin.

## Two evaluations, never blended

| Eval | Label | What it proves | Charts |
|------|--------|----------------|--------|
| **A** | `y_real` — matched cancellation covering ≥80% of invoice value | The pipeline predicts real returns on a held-out festive season | Untagged; primary Track 02 evidence |
| **B** | `y_rto` — Bernoulli draw from a documented logit | Product + ₹ cost + checkout action story | Every metric tagged **`[SCENARIO]`** |

The demo API (`POST /v1/risk/rto`) serves **Eval B** only.

## Data provenance

- **Primary source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (`id=502`), ~1.07M line items, 8 columns (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country).
- **Fetch:** `ucimlrepo` first, then UCI xlsx/zip URLs. Cached at `data/raw/online_retail_ii.parquet`.
- **Fallback:** If download fails, `src/data.py:make_synthetic_retail` builds a **schema-matched** toy set (same columns, C-invoices, guest checkouts, non-product codes). A `data/raw/SYNTHETIC_FALLBACK.txt` flag is written. Synthetic results are **not** UCI evidence.
- **Cleaning (deterministic):** drop `Price <= 0`; drop non-product stock codes (`POST`, `M`, `D`, `DOT`, `ADJUST*`, `CRUK`, `BANK CHARGES`, `AMAZONFEE`, `GIFT*`, …); drop null `Customer ID` for modeling (guest checkout has no history — drop rate is reported in EDA / `artifacts/metrics.json`).
- **`y_real` match:** each cancel line is greedily allocated to the **most recent prior** sales line with the same `(Customer ID, StockCode)`, `sales_date < cancel_date`, within 180 days. `return_value_frac = matched_credited_value / invoice_value`. **`y_real = 1` iff frac ≥ 0.80** (full-package analog of RTO). Partials are logged, not used as the primary label. Unmatched cancels are **not** attached to phantom orders.

## India overlay (Eval B only, `seed=42`)

UCI has no COD flag, no Indian PIN, no D2C fashion mix. Overlay fields are **synthetic and named as such**:

- **`amount_inr`:** percentile map of invoice GBP onto a D2C-like AOV (median ~₹1,100; p10~₹350; p90~₹3,500). **Not** an FX conversion of wholesale invoices.
- **`product_category`:** multinomial draw with fashion oversampled: fashion 40% / beauty 20% / home 25% / electronics 15%. Keyword flags on Description are **Eval A only**.
- **`pincode_tier` ∈ {metro, tier2, tier3}:** `Country` + `hash(customer_id)` into a **small static table of representative PIN prefixes** (Delhi/Mumbai/BLR/Chennai/Kolkata; Jaipur/Ahmedabad/Pune/Hyderabad/Lucknow; Hisar/Dehradun/Udaipur/Indore/Muzaffarpur). This is **not** a “bad area blocklist.”
- **`is_cod`:** ~60% base rate, higher for first-time and tier-3, lower for metro. Assigned **without reading `y_real`**.
- **`is_festive`:** invoice month ∈ {Nov, Dec}. Real UK peak used as a **proxy** for India’s festive spike — say so in every pitch.
- **`delivery_zone` / `days_to_expected_delivery`:** zone from PIN tier (local 2d / regional 4d / remote 7d) plus 0–1 day noise.

### Scenario label generator

```
logit = b0 + b_cod*COD + b_pin*pin_prior + b_fashion*fashion
      + b_festive*festive + b_hist*past_return_rate
      + 0.4*z_customer + 0.8*eps
y_rto ~ Bernoulli(sigmoid(logit))
```

- `pin_prior`: metro=0.0, tier2=0.40, tier3=0.80 (generator only).
- **`z_customer` and `eps` are not features.** They are the anti-AUC-theatre terms: a model that saw the full generator would overfit; a realistic scorer should land around **0.70–0.85 ROC AUC**, not 0.99.
- `past_return_rate` is the customer’s lagged `y_rto` (time-ordered). The model is *allowed* to see a lagged return rate feature; it is *not* allowed to see `z_customer` / `eps`.
- `b0` and `b_cod` are **calibrated empirically** so realized rates sit near prepaid ~5–7% and COD ~28–32%. Calibration is not “India truth.”

### Realized rates (filled after the first pipeline run)

See `artifacts/realized_rates.json` and `artifacts/metrics.json` — those files win if this paragraph drifts.

This run (`source=local:online_retail_II.xlsx`, n=36,604 invoices):

| Slice | Target band | Realized (this run) |
|-------|-------------|---------------------|
| Prepaid RTO | 5–7% | **7.00%** |
| COD RTO | 28–32% | **29.74%** |
| Fashion × COD | higher than COD overall | **35.14%** |
| Tier-3 × COD | higher than metro × COD | **36.78%** vs metro COD **22.85%** |
| COD share | ~60% mix | **62.3%** |
| Overall scenario rate | — | 21.16% |

Calibrated coefficients: `b0 ≈ -3.76`, `b_cod ≈ 1.91`.

## Published COD RTO ranges disagree (use a range, not a gospel number)

Industry write-ups do not agree with each other. Treat these as **directional context**:

- Shipway-style figures often cited around **~26% COD vs <2% prepaid**.
- Other logistics trackers commonly quote **~28–35% COD vs ~4–8% prepaid**.
- GoKwik vs Unicommerce **diverge at peak / festive** — the same week can be told as two different stories.
- RedSeer-style **₹ crore** industry-loss headlines are **pitch context only**, not a loss function we estimated.

This repo therefore **targets a band** (prepaid 5–7%, COD 28–32%) inside that disagreement, then **writes back realized simulator rates**. Do not present those realized rates as a census of Indian e-commerce.

## Cost priors (Eval B, INR)

Estimates for operating-point selection, not UCI-measured costs:

| Term | Default | Notes |
|------|---------|--------|
| **FN** (missed RTO) | **₹210** | Midpoint of published ~₹180–240 forward + reverse + handling, zero revenue |
| **FP** (genuine buyer flagged) | `0.25 * amount_inr * 0.15` | Margin 25%; conversion loss if pushed to prepaid 15% |
| FP sensitivity | 0.5× and 2× | Re-pick `t*` — show the operating point move |
| Optional verify step | ₹10, ~25% RTO cut among medium flags | **Assumption, not a measured lift** |

Eval A uses a **unitless proxy** (`C_fn=40`, `C_fp=1`) because there is no COD ₹ economics on UK giftware cancels. Break-even precision is 2.5%. Do not compare Eval A cost to Eval B rupees. Festive test contains only ~19 `y_real` positives — PR is high-variance; logistic AP beat XGB on this split, which we report rather than hide.

`total_cost(t) = FN_count(t)*C_fn + sum of per-order FP costs`. Threshold `t*` minimizes this on the **test** set (Nov–Dec 2011). That is an honest festive holdout, not a random shuffle.

## Split (no SMOTE, no random k-fold)

Return rates are **non-stationary**. A random k-fold would leak festive signal into training.

- Burn-in (history only): Dec 2009–Feb 2010
- Train: Mar 2010–Aug 2011
- Val / calibrate: Sep–Oct 2011
- **Test: Nov–Dec 2011** (festive holdout)

PIN-tier and category historical rates are fit on **train orders only**, then mapped. Customer features are lagged and exclude the current invoice. `scale_pos_weight = n_neg/n_pos` on **actual training prevalence**.

## Fairness

PIN-tier scores can proxy **caste / religion / class geography**. PIN is a feature for **verification routing**, not a refusal rule. `artifacts/metrics.json` reports FP rate by PIN tier on test. If tier-3 FP is much higher than metro, that is called out. Demo copy is “extra verification,” never “we don’t ship to this PIN.”

## Defense-only actions

| Tier | Rule | Action |
|------|------|--------|
| Low (`p < t_med`) | Proceed | COD allowed |
| Medium | Human loop | Confirmation call / SMS / WhatsApp **before dispatch** |
| High (`p ≥ t*`) | Friction | Require **prepaid or address re-verification** |
| — | Forbidden | Silent cancel, auto-block, region ban, `denied` |

Unknown `customer_id` / missing features → conservative **medium** + `reason=insufficient_history`, still no deny.

## What this is not

- Not a post-delivery returns model, not chargebacks, not Razorpay live production.
- Not Olist, not SMOTE, not k-fold CV.
- Not a claim that UCI cancellations equal Indian RTO.
