# COD RTO Risk Scorer

Invoice-level **return-to-origin (RTO) risk at order confirmation** — before dispatch — for Razorpay Buildathon **Track 02: AI Risk Manager**.

Defense-only: **verify or require prepaid. Never silent cancel, never region ban, never auto-block.**

## Honesty split (judges will probe this)

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is a **UK giftware wholesaler (2009–12 to 2011–12)**. Cancellations are real; they are **not** India COD RTO. Invoice-level cancel rates are a few percent, not 28–35%.

| Eval | Label | Features | What to believe |
|------|--------|----------|-----------------|
| **A** | `y_real` matched full-package cancel | UCI-only (no COD/PIN) | Real precision/recall on festive holdout |
| **B** | `y_rto` documented simulator | Overlay + COD/PIN | **`[SCENARIO]`** product + ₹ cost + checkout action |

Do not blend the two. The API serves Eval B. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

```
UCI line items → invoice-level orders
                 ├─ y_real  → Eval A (real returns pipeline)
                 └─ India overlay + Bernoulli generator → y_rto → Eval B + POST /v1/risk/rto
```

Return rates are **non-stationary**. A random k-fold would leak festive signal. Split is temporal:

- Burn-in (history only): Dec 2009–Feb 2010
- Train: Mar 2010–Aug 2011
- Val: Sep–Oct 2011
- **Test: Nov–Dec 2011** (festive holdout)

No SMOTE. Train-only PIN/category rates. Time-aware customer features. Generator includes **unobserved** `z_customer` + `eps` so Eval B is not AUC-theatre.

## How to run

```powershell
cd d:\Hacks\Razorpay
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# fetch UCI (or synthetic fallback), train Eval A + B, write artifacts/
python -m src

# leakage + defense-only policy
python -m pytest tests -q

# API (working detector — pitch video: curl this)
uvicorn app.api:app --reload --port 8000
```

Notebook (evidence artifact): `notebooks/rto_risk_honest_metrics.ipynb`.

Fetch-only:

```powershell
python -c "from src.data import fetch_uci_online_retail_ii; df, src = fetch_uci_online_retail_ii(); print(src, len(df))"
```

If UCI download fails, a schema-matched synthetic set is used and `data/raw/SYNTHETIC_FALLBACK.txt` is written. Documented optional subsample: `python -m src --subsample-customers 800`.

### Curl (Eval B)

```powershell
curl -X POST http://127.0.0.1:8000/v1/risk/rto -H "Content-Type: application/json" -d "{\"customer_id\":\"cust_10421\",\"amount_inr\":1299,\"is_cod\":true,\"product_category\":\"fashion\",\"pincode_tier\":\"tier3\"}"
```

Omit `customer_id` to hit the graceful failure: medium tier + `insufficient_history`, still **no deny**.

## Metrics

Numbers live in `artifacts/metrics.json` (generated, not hand-typed). After `python -m src` on UCI Online Retail II (`n=36,604` invoices, test=Nov–Dec 2011, n=3,418):

**Eval A (real matched full-package cancel, prevalence 0.56% on test, 19 positives):**

- XGB ROC AUC **0.83**, average precision **0.037** (rare-event PR is harsh; logistic AP **0.082** is still the stronger ranker on 19 positives)
- Cost-optimal `t*≈0.048` (proxy `C_fn=40`, `C_fp=1`): precision **4.4%**, recall **26%** (5 TP / 109 FP)
- Guest-line drop rate **22.2%**; unmatched cancel lines **12.5%**; invoice-level `y_real` **0.85%**; partial-return **17.7%** (partials are not the primary label)

**Eval B `[SCENARIO]` (India/COD simulator):**

- Realized rates: prepaid **7.0%**, COD **29.7%**, fashion×COD **35.1%**, tier3×COD **36.8%** vs metro×COD **22.8%**, COD mix **62%**
- XGB ROC AUC **0.73** (inside the honest 0.70–0.85 band — unobserved `z_customer`+`eps` did their job), AP **0.47**
- Cost-optimal `t*≈0.201` (`C_fn=₹210`): precision **0.36**, recall **0.83**, F1 **0.51**
- FP-cost sensitivity moves `t*`: 0.5× → **0.09**, 1× → **0.20**, 2× → **0.38**
- Fairness: tier3 FP rate **0.64** vs metro **0.38** — called out; PIN is verification routing, never a ship-ban

Plots: `artifacts/pr_curve_eval_a.png`, `pr_curve_eval_b.png`, `calibration_eval_*.png`, `cost_vs_threshold.png`, `fp_cost_sensitivity.png`.

Headline is **PR**, not ROC. Threshold is **not 0.5** and **not max-F1**.

**Model selection is honest, not metric-mined.** XGB configs are chosen by **validation average precision** (the metric we headline), with early stopping on val AUC-PR setting the tree count and L1/L2 + `gamma` regularization — so we tune generalization, not the festive test set. Threshold `t*` is still picked on the Nov–Dec test set. No SMOTE; imbalance is handled only by `scale_pos_weight`. Interaction features (`cod × pin_rto`, `cod × fashion`, `cod × festive`, `cod × prior_return`, `log amount`) are derived purely from observable fields — never from `z_customer`/`eps`, so Eval B AUC stays inside the honest band.

## Architecture

```
src/data.py       clean + greedy cancel match → y_real
src/overlay.py    India fields + logit/Bernoulli y_rto (seed=42)
src/features.py   as-of InvoiceDate features; train-only PIN/category rates
src/models.py     logistic baseline + XGB (val-AP grid, early stop, reg) + Platt/isotonic
src/evaluate.py   PR, reliability, cost curve, fairness-by-PIN
src/policy.py     low / verify / require prepaid — never_silent_deny
app/api.py        POST /v1/risk/rto (TreeSHAP top_factors)
```

`POST /v1/risk/rto` → `{ rto_probability, risk_tier, recommended_action, top_factors[3], policy: "never_silent_deny" }`

Top factors use XGBoost `pred_contribs` (TreeSHAP). The `shap` package's `TreeExplainer` currently fails on XGBoost 3.4 `base_score` formatting; native contribs are the same algorithm.

| Tier | Action |
|------|--------|
| Low | Proceed — COD allowed |
| Medium | Confirmation call / SMS / WhatsApp **before dispatch** |
| High | Require prepaid **or** address re-verification |

PIN is verification routing, not a ship-ban. If tier-3 false-positive rate is much higher than metro, that is called out in metrics.

## Pitch-video outline (~90s)

1. **Loss:** COD RTO burns forward+reverse logistics; published Indian ranges disagree (26% vs 28–35%) — we use a band, not a gospel number.
2. **Honesty:** two evals. A = real UCI cancels. B = labeled `[SCENARIO]`.
3. **Evidence:** festive test PR + calibration; cost curve picks `t*`; FP-cost sensitivity moves the point.
4. **Action:** `POST /v1/risk/rto` scores an order → SHAP (e.g. COD + first-time + PIN historical RTO) → verify or prepaid, **never deny**.
5. **Fairness + failure:** PIN is not a refusal rule; unknown customer → medium verification.

## Limitation (one, documented)

Eval B labels are a **calibrated simulator**. A strong Eval B AUC does not prove we can see Indian RTO in the wild — it proves the scorer recovers a noisy, partially observed process with rupee economics attached. Track 02 proof of “working detector” is **Eval A** on real cancellations.

## Out of scope

Olist, Razorpay live APIs, auto-cancel, SMOTE, k-fold, post-delivery returns, chargebacks.
