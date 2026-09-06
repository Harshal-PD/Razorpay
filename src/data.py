"""Load, clean, and label UCI Online Retail II at invoice level.

Honesty: cancellations are real UK giftware events, not India COD RTO.
`y_real` is a matched full-package cancel analog (return_value_frac >= 0.80).
"""

from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    MATCH_WINDOW_DAYS,
    NON_PRODUCT_STOCK,
    RANDOM_SEED,
    UCI_ID,
    UCI_XLSX_URLS,
    Y_REAL_FRAC_THRESHOLD,
)

LINE_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        low = key.lower().replace("_", " ")
        if low in {"invoiceno", "invoice no", "invoice"}:
            rename[c] = "Invoice"
        elif low in {"stockcode", "stock code"}:
            rename[c] = "StockCode"
        elif low == "description":
            rename[c] = "Description"
        elif low == "quantity":
            rename[c] = "Quantity"
        elif low in {"invoicedate", "invoice date"}:
            rename[c] = "InvoiceDate"
        elif low in {"unitprice", "unit price", "price"}:
            rename[c] = "Price"
        elif low in {"customerid", "customer id"}:
            rename[c] = "Customer ID"
        elif low == "country":
            rename[c] = "Country"
    out = df.rename(columns=rename)
    missing = [c for c in LINE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Unexpected schema, missing {missing}. Have {list(out.columns)}")
    out = out[LINE_COLUMNS].copy()
    out["Invoice"] = out["Invoice"].astype(str)
    out["StockCode"] = out["StockCode"].astype(str)
    out["Description"] = out["Description"].astype(str)
    out["Country"] = out["Country"].astype(str)
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out["Customer ID"] = pd.to_numeric(out["Customer ID"], errors="coerce")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    return out


def _is_non_product(code: Any) -> bool:
    c = str(code).strip().upper()
    if c in NON_PRODUCT_STOCK:
        return True
    if c.startswith("ADJUST"):
        return True
    if c.startswith("GIFT"):
        return True
    if c.startswith("AMAZON"):
        return True
    if "BANK" in c:
        return True
    if c.startswith("DCGS"):
        return True
    if c.startswith("DOT"):
        return True
    return False


def make_synthetic_retail(n_customers: int = 900, n_invoices: int = 14000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Schema-matched fallback if UCI cannot be downloaded. Documented in ASSUMPTIONS.md."""
    rng = np.random.default_rng(seed)
    start = datetime(2009, 12, 1)
    end = datetime(2011, 12, 9)
    span_days = (end - start).days

    stock_codes = [f"{10000 + i}" for i in range(80)] + ["POST", "M", "D", "DOT"]
    countries = ["United Kingdom"] * 18 + ["France", "Germany", "EIRE", "Netherlands"]
    descriptions = {
        "POST": "POSTAGE",
        "M": "Manual",
        "D": "Discount",
        "DOT": "DOTCOM POSTAGE",
    }
    gift_words = [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "JUMBO BAG RED RETROSPOT",
        "REGENCY CAKESTAND 3 TIER",
        "PARTY BUNTING",
        "LUNCH BAG BLACK SKULL",
        "SET OF 3 CAKE TINS PANTRY DESIGN",
        "PACK OF 72 RETROSPOT CAKE CASES",
        "NATURAL SLATE HEART CHALKBOARD",
        "HEART OF WICKER SMALL",
        "CHOCOLATE HOT WATER BOTTLE",
        "VINTAGE SNAKES & LADDERS",
        "FELTCRAFT DOLL MARIA",
        "RED WOOLLY HOTTIE WHITE HEART",
        "GARDENERS KNEELING PAD CUP OF TEA",
        "ALARM CLOCK BAKELIKE RED",
    ]

    customers = np.arange(13000, 13000 + n_customers).astype(float)
    rows: list[dict[str, Any]] = []
    invoice_no = 490000
    # Guest invoices (null customer) for drop-rate reporting
    n_guest = max(40, n_invoices // 20)

    def _add_invoice(inv: str, dt: datetime, cid: Any, n_lines: int, country: str, qty_sign: int) -> None:
        used = rng.choice(stock_codes[:80], size=n_lines, replace=False)
        for sc in used:
            qty = int(qty_sign * rng.integers(1, 8))
            price = float(np.round(rng.uniform(0.85, 18.0), 2))
            desc = gift_words[hash(str(sc)) % len(gift_words)]
            rows.append(
                {
                    "Invoice": inv,
                    "StockCode": sc,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": dt,
                    "Price": price,
                    "Customer ID": cid,
                    "Country": country,
                }
            )

    # Sales invoices
    sale_meta: list[tuple[str, datetime, float, str, list[str]]] = []
    for i in range(n_invoices):
        dt = start + timedelta(days=int(rng.integers(0, span_days)), hours=int(rng.integers(8, 18)))
        cid = float(rng.choice(customers))
        country = str(rng.choice(countries))
        inv = str(invoice_no)
        invoice_no += 1
        n_lines = int(rng.integers(1, 6))
        _add_invoice(inv, dt, cid, n_lines, country, 1)
        sale_meta.append((inv, dt, cid, country, []))

    # Matched cancels (~8% of sales get a later C invoice for some lines)
    n_cancel = int(n_invoices * 0.08)
    pick = rng.choice(len(sale_meta), size=n_cancel, replace=False)
    sales_df_tmp = pd.DataFrame(rows)
    for idx in pick:
        inv, dt, cid, country, _ = sale_meta[idx]
        lines = sales_df_tmp[sales_df_tmp["Invoice"] == inv]
        if lines.empty:
            continue
        lag = int(rng.integers(1, 40))
        cdt = dt + timedelta(days=lag)
        if cdt > end + timedelta(days=20):
            continue
        cinv = "C" + str(invoice_no)
        invoice_no += 1
        take = lines.sample(n=min(len(lines), int(rng.integers(1, len(lines) + 1))), random_state=int(rng.integers(1e9)))
        # ~70% full-invoice cancel analog (all lines), rest partial
        if rng.random() < 0.55:
            take = lines
        for r in take.itertuples(index=False):
            rows.append(
                {
                    "Invoice": cinv,
                    "StockCode": r.StockCode,
                    "Description": r.Description,
                    "Quantity": -abs(int(r.Quantity)),
                    "InvoiceDate": cdt,
                    "Price": r.Price,
                    "Customer ID": cid,
                    "Country": country,
                }
            )

    # Guest checkouts
    for _ in range(n_guest):
        dt = start + timedelta(days=int(rng.integers(0, span_days)))
        inv = str(invoice_no)
        invoice_no += 1
        rows.append(
            {
                "Invoice": inv,
                "StockCode": str(rng.choice(stock_codes[:80])),
                "Description": rng.choice(gift_words),
                "Quantity": int(rng.integers(1, 4)),
                "InvoiceDate": dt,
                "Price": float(np.round(rng.uniform(1.0, 12.0), 2)),
                "Customer ID": np.nan,
                "Country": "United Kingdom",
            }
        )

    # A few non-product and zero-price rows
    rows.append(
        {
            "Invoice": str(invoice_no),
            "StockCode": "POST",
            "Description": "POSTAGE",
            "Quantity": 1,
            "InvoiceDate": start + timedelta(days=10),
            "Price": 18.0,
            "Customer ID": float(customers[0]),
            "Country": "United Kingdom",
        }
    )
    rows.append(
        {
            "Invoice": str(invoice_no + 1),
            "StockCode": "10002",
            "Description": "ZERO PRICE",
            "Quantity": 1,
            "InvoiceDate": start + timedelta(days=11),
            "Price": 0.0,
            "Customer ID": float(customers[1]),
            "Country": "United Kingdom",
        }
    )

    df = pd.DataFrame(rows)
    df.attrs["source"] = "synthetic_fallback"
    return df


def _download_url(url: str, dest: Path, timeout: int = 45) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    import socket
    import subprocess

    curl = shutil.which("curl") or shutil.which("curl.exe")
    tmp = dest.with_suffix(dest.suffix + ".part")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "-C",
            "-",
            "--max-time",
            str(timeout),
            "-A",
            "Mozilla/5.0 (compatible; cod-rto-risk-scorer/0.1)",
            "-o",
            str(tmp),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            raise RuntimeError(f"curl failed ({proc.returncode}): {proc.stderr[-400:]}")
        data = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
    else:
        socket.setdefaulttimeout(timeout)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; cod-rto-risk-scorer/0.1)"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()

    if url.endswith(".zip") or data[:2] == b"PK":
        zpath = dest.with_suffix(".zip")
        zpath.write_bytes(data)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))]
            if not names:
                raise RuntimeError(f"No tabular file in zip from {url}")
            inner = names[0]
            out = dest.parent / Path(inner).name
            out.write_bytes(zf.read(inner))
            return out
    dest.write_bytes(data)
    return dest


def _read_uci_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    return _normalize_columns(df)


def _fetch_ucimlrepo_timeout(seconds: int) -> pd.DataFrame:
    import threading

    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            from ucimlrepo import fetch_ucirepo

            bundle = fetch_ucirepo(id=UCI_ID)
            raw = getattr(bundle.data, "original", None)
            if raw is None:
                raw = pd.concat([bundle.data.features, bundle.data.targets], axis=1)
            holder["df"] = _normalize_columns(pd.DataFrame(raw))
        except Exception as exc:  # noqa: BLE001
            holder["err"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=seconds)
    if t.is_alive():
        raise TimeoutError(f"ucimlrepo fetch_ucirepo exceeded {seconds}s")
    if "err" in holder:
        raise holder["err"]
    if "df" not in holder:
        raise RuntimeError("ucimlrepo returned no frame")
    return holder["df"]


def fetch_uci_online_retail_ii(cache_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Return (line_items, source_tag). Tries local/xlsx/zip, then timed ucimlrepo, then synthetic."""
    cache_dir = cache_dir or DATA_RAW
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_cache = cache_dir / "online_retail_ii.parquet"
    large_local = [
        p
        for p in cache_dir.glob("*")
        if p.suffix.lower() in {".xlsx", ".xls"} and p.stat().st_size > 5_000_000
    ]

    if parquet_cache.exists():
        df = pd.read_parquet(parquet_cache)
        if len(df) >= 200_000:
            print(f"Using cached {parquet_cache} ({len(df):,} rows)", flush=True)
            src = str(df.attrs.get("source", "parquet_cache"))
            return _normalize_columns(df), src
        print(f"Ignoring small parquet cache ({len(df):,} rows)", flush=True)

    errors: list[str] = []

    for local in sorted(large_local) or sorted(cache_dir.glob("*")):
        if local.suffix.lower() in {".xlsx", ".xls", ".csv"} and local.stat().st_size > 1000:
            try:
                print(f"Reading local file {local}", flush=True)
                df = _read_uci_file(local)
                df.attrs["source"] = f"local:{local.name}"
                df.to_parquet(parquet_cache, index=False)
                return df, f"local:{local.name}"
            except Exception as exc:  # noqa: BLE001
                print(f"Local file failed {local}: {exc!r}", flush=True)
                errors.append(f"local {local.name}: {exc!r}")

    for url in UCI_XLSX_URLS:
        try:
            print(f"Downloading {url} ...", flush=True)
            dest = cache_dir / "online_retail_II.xlsx"
            path = _download_url(url, dest, timeout=180)
            df = _read_uci_file(path)
            df.attrs["source"] = f"xlsx:{url}"
            df.to_parquet(parquet_cache, index=False)
            print(f"Loaded {len(df):,} rows from {url}", flush=True)
            return df, f"xlsx:{url}"
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc!r}", flush=True)
            errors.append(f"{url}: {exc!r}")

    uci_err: Exception | str = "skipped"
    try:
        print("Trying ucimlrepo id=502 with 60s timeout ...", flush=True)
        df = _fetch_ucimlrepo_timeout(60)
        df.attrs["source"] = "ucimlrepo:502"
        df.to_parquet(parquet_cache, index=False)
        print(f"Loaded {len(df):,} rows from ucimlrepo", flush=True)
        return df, "ucimlrepo:502"
    except Exception as exc:  # noqa: BLE001
        uci_err = exc
        errors.append(f"ucimlrepo: {exc!r}")
        print(f"ucimlrepo failed: {exc!r}", flush=True)

    print("UCI download failed; using schema-matched synthetic fallback.", flush=True)
    df = make_synthetic_retail()
    df.attrs["source"] = "synthetic_fallback"
    note = cache_dir / "SYNTHETIC_FALLBACK.txt"
    note.write_text(
        "UCI Online Retail II download failed; using schema-matched synthetic fallback.\n"
        f"ucimlrepo error: {uci_err!r}\n"
        "Errors:\n" + "\n".join(errors) + "\n"
        "This is NOT the real UCI dataset. See ASSUMPTIONS.md.\n",
        encoding="utf-8",
    )
    df.to_parquet(parquet_cache, index=False)
    return df, "synthetic_fallback"


def clean_line_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic cleaning. Returns cleaned frame + drop stats for EDA."""
    n0 = len(df)
    out = df.copy()
    out["Invoice"] = out["Invoice"].astype(str)
    out["StockCode"] = out["StockCode"].astype(str)
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Customer ID"] = pd.to_numeric(out["Customer ID"], errors="coerce")

    price_nonpos = int((out["Price"] <= 0).sum())
    out = out.loc[out["Price"] > 0]
    non_product = out["StockCode"].map(_is_non_product)
    n_non_product = int(non_product.sum())
    out = out.loc[~non_product]

    n_guest = int(out["Customer ID"].isna().sum())
    guest_rate = float(n_guest / max(len(out), 1))
    modeled = out.loc[out["Customer ID"].notna()].copy()
    modeled["Customer ID"] = modeled["Customer ID"].astype(int)

    stats = {
        "n_raw": n0,
        "n_price_nonpos_dropped": price_nonpos,
        "n_non_product_dropped": n_non_product,
        "n_after_product_filter": int(len(out) + n_non_product),  # before guest drop, after price/stock
        "n_guest_lines": n_guest,
        "guest_line_drop_rate": guest_rate,
        "n_modeled_lines": int(len(modeled)),
        "n_unique_customers": int(modeled["Customer ID"].nunique()),
    }
    # fix n_after: lines after price+stock filter including guests
    stats["n_after_price_stock"] = int(len(out))
    return modeled, stats


def split_sales_cancels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = df["Invoice"].astype(str)
    is_cancel = inv.str.startswith("C")
    sales = df.loc[~is_cancel & (df["Quantity"] > 0)].copy()
    cancels = df.loc[is_cancel & (df["Quantity"] < 0)].copy()
    return sales, cancels


def match_cancels_to_sales(
    sales: pd.DataFrame,
    cancels: pd.DataFrame,
    window_days: int = MATCH_WINDOW_DAYS,
) -> tuple[pd.Series, dict[str, Any]]:
    """Greedy most-recent prior (Customer ID, StockCode) match within window; allocate qty.

    Returns credited value keyed by **sales Invoice**, plus match stats.
    Unmatched cancels are not attached to phantom orders.
    """
    sales_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    s = sales.rename(columns={"Customer ID": "cid"})
    c = cancels.rename(columns={"Customer ID": "cid"})

    for r in s.itertuples(index=False):
        sales_by_key[(int(r.cid), str(r.StockCode))].append(
            {
                "invoice": str(r.Invoice),
                "date": pd.Timestamp(r.InvoiceDate),
                "remaining": float(r.Quantity),
                "price": float(r.Price),
            }
        )
    for lst in sales_by_key.values():
        lst.sort(key=lambda x: x["date"])

    credited: dict[str, float] = defaultdict(float)
    n_cancel_lines = 0
    n_matched_lines = 0
    n_unmatched_lines = 0
    matched_qty = 0.0
    unmatched_qty = 0.0

    c_sorted = c.sort_values("InvoiceDate")
    window = pd.Timedelta(days=window_days)
    for r in c_sorted.itertuples(index=False):
        n_cancel_lines += 1
        need = float(abs(r.Quantity))
        key = (int(r.cid), str(r.StockCode))
        cdate = pd.Timestamp(r.InvoiceDate)
        lines = sales_by_key.get(key)
        if not lines:
            n_unmatched_lines += 1
            unmatched_qty += need
            continue
        # walk most-recent prior: from end, skip sales on/after cancel date or outside window
        got = 0.0
        for rec in reversed(lines):
            if rec["remaining"] <= 0:
                continue
            if rec["date"] >= cdate:
                continue
            if cdate - rec["date"] > window:
                continue
            take = min(rec["remaining"], need - got)
            rec["remaining"] -= take
            credited[rec["invoice"]] += take * rec["price"]
            got += take
            if got >= need - 1e-9:
                break
        if got > 1e-9:
            n_matched_lines += 1
            matched_qty += got
            leftover = need - got
            if leftover > 1e-9:
                unmatched_qty += leftover
        else:
            n_unmatched_lines += 1
            unmatched_qty += need

    stats = {
        "n_cancel_lines": n_cancel_lines,
        "n_matched_cancel_lines": n_matched_lines,
        "n_unmatched_cancel_lines": n_unmatched_lines,
        "matched_qty": matched_qty,
        "unmatched_qty": unmatched_qty,
        "unmatched_cancel_line_rate": float(n_unmatched_lines / max(n_cancel_lines, 1)),
    }
    return pd.Series(credited, name="matched_credited_value"), stats


def build_invoice_table(
    sales: pd.DataFrame,
    credited: pd.Series,
    y_threshold: float = Y_REAL_FRAC_THRESHOLD,
) -> pd.DataFrame:
    """One row per sales invoice with y_real (full-package cancel analog)."""
    tmp = sales.copy()
    tmp["line_value"] = tmp["Quantity"] * tmp["Price"]
    desc = (
        tmp.groupby("Invoice")["Description"]
        .apply(lambda s: " ".join(s.dropna().astype(str).head(6)))
        .rename("description")
    )
    orders = tmp.groupby("Invoice", as_index=False).agg(
        invoice_date=("InvoiceDate", "min"),
        customer_id=("Customer ID", "first"),
        country=("Country", "first"),
        qty=("Quantity", "sum"),
        n_lines=("StockCode", "count"),
        amount_gbp=("line_value", "sum"),
    )
    orders["matched_credited_value"] = orders["Invoice"].map(credited).fillna(0.0)
    orders["return_value_frac"] = orders["matched_credited_value"] / orders["amount_gbp"].clip(lower=1e-9)
    orders["y_real"] = (orders["return_value_frac"] >= y_threshold).astype(int)
    orders["is_partial_return"] = (
        (orders["return_value_frac"] > 0) & (orders["return_value_frac"] < y_threshold)
    ).astype(int)
    orders = orders.merge(desc, left_on="Invoice", right_index=True, how="left")
    orders = orders.rename(columns={"Invoice": "invoice"})
    orders["customer_id"] = orders["customer_id"].astype(int)
    return orders


def load_orders(use_cache: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full path: fetch → clean → match → invoice table. Caches processed parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = DATA_PROCESSED / "orders.parquet"
    meta_path = DATA_PROCESSED / "clean_stats.json"
    if use_cache and cache.exists():
        orders = pd.read_parquet(cache)
        stats = {}
        if meta_path.exists():
            import json

            stats = json.loads(meta_path.read_text(encoding="utf-8"))
        src = str(stats.get("source", ""))
        if len(orders) >= 20000 or src.startswith("local:") or src.startswith("xlsx") or src.startswith("ucimlrepo"):
            return orders, stats
        print(f"Ignoring small/synthetic processed cache ({len(orders)} rows, source={src})", flush=True)

    lines, source = fetch_uci_online_retail_ii()
    print(f"Fetched {len(lines):,} line items from {source}", flush=True)
    modeled, clean_stats = clean_line_items(lines)
    print(f"Cleaned modeled lines={len(modeled):,}", flush=True)
    sales, cancels = split_sales_cancels(modeled)
    print(f"Matching {len(cancels):,} cancel lines to {len(sales):,} sales lines ...", flush=True)
    credited, match_stats = match_cancels_to_sales(sales, cancels)
    orders = build_invoice_table(sales, credited)
    print(f"Invoice table n={len(orders):,} y_real_rate={float(orders['y_real'].mean()):.4f}", flush=True)
    stats = {
        "source": source,
        "clean": clean_stats,
        "match": match_stats,
        "n_sales_invoices": int(len(orders)),
        "y_real_rate": float(orders["y_real"].mean()),
        "partial_return_rate": float(orders["is_partial_return"].mean()),
        "n_y_real_pos": int(orders["y_real"].sum()),
    }
    orders.attrs["source"] = source
    orders.to_parquet(cache, index=False)
    import json

    meta_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return orders, stats
