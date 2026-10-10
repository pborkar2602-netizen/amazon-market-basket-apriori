"""
FastAPI application for Basketry: Market Basket Analysis on Amazon Product Reviews.

Serves:
- REST API endpoints for dataset upload, association rule mining, algorithm comparison,
  parameter sweeping, manual metric derivations, and product bundle recommendations.
- Static file serving for the custom HTML/CSS/JS frontend.
"""

import os
import io
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np

from app.preprocess import (
    detect_columns,
    summarize_dataset,
    clean_and_prepare_baskets,
)
from app.apriori_scratch import apriori_from_scratch
from app.algorithms import (
    run_mlxtend_apriori,
    run_mlxtend_fpgrowth,
    execute_mining,
)
from app.metrics import format_manual_worked_example

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEMO_FILE = os.path.join(DATA_DIR, "demo_small.csv")

from contextlib import asynccontextmanager

class MineParams(BaseModel):
    algorithm: str = Field(default="scratch", description="'scratch', 'apriori', or 'fpgrowth'")
    min_support: float = Field(default=0.03, ge=0.001, le=1.0)
    min_confidence: float = Field(default=0.35, ge=0.01, le=1.0)
    min_lift: float = Field(default=1.0, ge=0.0)
    min_rating: float = Field(default=4.0, ge=1.0, le=5.0)
    max_len: int = Field(default=3, ge=1, le=5)


class AppState:
    """Manages active dataset, preprocessed transaction cache, and mining results."""

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.transactions: List[List[str]] = []
        self.one_hot_df: pd.DataFrame = pd.DataFrame()
        self.title_map: Dict[str, str] = {}
        self.meta: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}
        self.active_source: str = "demo_small.csv"
        self.current_min_rating: float = 4.0
        self.mining_cache: Dict[str, Any] = {}
        self.latest_rules: List[Dict[str, Any]] = []

    def load_csv_data(self, df: pd.DataFrame, source_name: str, min_rating: float = 4.0):
        col_map = detect_columns(list(df.columns))
        if not col_map.get("reviewer") or not col_map.get("product"):
            raise ValueError(
                "Unable to detect reviewer and product columns. "
                "Ensure headers contain 'reviewerID' (or 'user') and 'asin' (or 'product')."
            )

        self.raw_df = df
        self.active_source = source_name
        self.current_min_rating = min_rating
        self.summary = summarize_dataset(df, col_map)

        transactions, one_hot_df, title_map, meta = clean_and_prepare_baskets(
            df=df,
            min_rating=min_rating,
            col_map=col_map,
        )

        self.transactions = transactions
        self.one_hot_df = one_hot_df
        self.title_map = title_map
        self.meta = meta
        self.mining_cache.clear()
        self.latest_rules.clear()

    def update_rating_threshold(self, min_rating: float):
        if self.raw_df is not None and abs(self.current_min_rating - min_rating) > 1e-4:
            self.current_min_rating = min_rating
            col_map = self.summary.get("detected_columns", detect_columns(list(self.raw_df.columns)))
            transactions, one_hot_df, title_map, meta = clean_and_prepare_baskets(
                df=self.raw_df,
                min_rating=min_rating,
                col_map=col_map,
            )
            self.transactions = transactions
            self.one_hot_df = one_hot_df
            self.title_map = title_map
            self.meta = meta
            self.mining_cache.clear()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load demo_small.csv immediately so the app is instant and ready on launch."""
    if os.path.exists(DEMO_FILE):
        try:
            df = pd.read_csv(DEMO_FILE)
            state.load_csv_data(df, "demo_small.csv", min_rating=4.0)
            print(f"[startup] Loaded demo_small.csv ({len(df):,} reviews, {len(state.transactions):,} baskets)")
        except Exception as e:
            print(f"[startup] Error loading demo_small.csv: {e}")
    yield

app = FastAPI(
    title="Basketry",
    description="Market Basket Analysis on Amazon Product Reviews using Association Rule Mining",
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/summary")
def get_summary():
    """Return current dataset summary and preprocessing metadata."""
    if state.raw_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    return {
        "source": state.active_source,
        "dataset_summary": state.summary,
        "preprocessed_meta": state.meta,
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept an uploaded CSV file, auto-detect reviewer, product and rating columns,
    and update current dataset state.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    try:
        contents = await file.read()
        # Decode UTF-8 or Latin-1
        try:
            text_stream = io.StringIO(contents.decode("utf-8"))
        except UnicodeDecodeError:
            text_stream = io.StringIO(contents.decode("latin-1"))

        df = pd.read_csv(text_stream)

        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

        state.load_csv_data(df, file.filename, min_rating=state.current_min_rating)

        return {
            "status": "success",
            "message": f"Successfully loaded '{file.filename}' with {len(df):,} reviews.",
            "source": file.filename,
            "dataset_summary": state.summary,
            "preprocessed_meta": state.meta,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.post("/api/sample")
def load_sample_dataset(name: str = Query("demo", description="'demo' (2k reviewers) or 'sample' (30k reviewers)")):
    """Reload one of the built-in datasets."""
    file_target = DEMO_FILE if name == "demo" else os.path.join(DATA_DIR, "reviews_sample.csv")
    if not os.path.exists(file_target):
        file_target = DEMO_FILE

    try:
        df = pd.read_csv(file_target)
        state.load_csv_data(df, os.path.basename(file_target), min_rating=state.current_min_rating)
        return {
            "status": "success",
            "source": state.active_source,
            "dataset_summary": state.summary,
            "preprocessed_meta": state.meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading sample data: {str(e)}")


@app.post("/api/mine")
def mine_rules(params: MineParams):
    """
    Mine frequent itemsets and association rules using specified algorithm:
    'scratch' (Custom Apriori), 'apriori' (mlxtend), or 'fpgrowth' (mlxtend).
    """
    if state.raw_df is None or not state.transactions:
        raise HTTPException(status_code=400, detail="No qualifying transaction baskets available.")

    # Check if rating threshold changed
    state.update_rating_threshold(params.min_rating)

    cache_key = (
        f"{params.algorithm}_{params.min_support}_{params.min_confidence}_"
        f"{params.min_lift}_{params.min_rating}_{params.max_len}"
    )

    if cache_key in state.mining_cache:
        cached_result = state.mining_cache[cache_key]
        state.latest_rules = cached_result["rules"]
        return cached_result

    try:
        result = execute_mining(
            algorithm=params.algorithm,
            transactions=state.transactions,
            one_hot_df=state.one_hot_df,
            min_support=params.min_support,
            min_confidence=params.min_confidence,
            min_lift=params.min_lift,
            max_len=params.max_len,
            title_map=state.title_map,
        )

        state.mining_cache[cache_key] = result
        state.latest_rules = result["rules"]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mining failed: {str(e)}")


@app.get("/api/compare")
def compare_algorithms(
    min_support: float = Query(0.03, ge=0.001, le=1.0),
    min_confidence: float = Query(0.35, ge=0.01, le=1.0),
    min_lift: float = Query(1.0, ge=0.0),
    min_rating: float = Query(4.0, ge=1.0, le=5.0),
    max_len: int = Query(3, ge=1, le=5),
):
    """
    Run all three algorithms (Scratch Apriori, mlxtend Apriori, mlxtend FP-Growth)
    on the exact same data and parameters.
    Measures runtimes and mathematically verifies that Scratch Apriori matches mlxtend.
    """
    if state.raw_df is None or not state.transactions:
        raise HTTPException(status_code=400, detail="No transaction baskets available.")

    state.update_rating_threshold(min_rating)

    # 1. Scratch Apriori
    res_scratch = apriori_from_scratch(
        transactions=state.transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=state.title_map,
    )

    # 2. mlxtend Apriori
    res_apriori = run_mlxtend_apriori(
        one_hot_df=state.one_hot_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=state.title_map,
    )

    # 3. mlxtend FP-Growth
    res_fpgrowth = run_mlxtend_fpgrowth(
        one_hot_df=state.one_hot_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=state.title_map,
    )

    # Verification: set comparison
    scratch_itemsets = {tuple(sorted(x["itemset"])) for x in res_scratch["itemsets"]}
    mlx_itemsets = {tuple(sorted(x["itemset"])) for x in res_apriori["itemsets"]}
    fp_itemsets = {tuple(sorted(x["itemset"])) for x in res_fpgrowth["itemsets"]}

    itemsets_match = (scratch_itemsets == mlx_itemsets == fp_itemsets)

    scratch_rules = {
        (tuple(sorted(r["antecedent"])), tuple(sorted(r["consequent"])))
        for r in res_scratch["rules"]
    }
    mlx_rules = {
        (tuple(sorted(r["antecedent"])), tuple(sorted(r["consequent"])))
        for r in res_apriori["rules"]
    }
    rules_match = (scratch_rules == mlx_rules)

    return {
        "status": "success",
        "verification": {
            "itemsets_match": itemsets_match,
            "rules_match": rules_match,
            "scratch_itemset_count": len(scratch_itemsets),
            "mlx_itemset_count": len(mlx_itemsets),
            "fp_itemset_count": len(fp_itemsets),
            "scratch_rule_count": len(scratch_rules),
            "mlx_rule_count": len(mlx_rules),
        },
        "scratch": {
            "algorithm": "Apriori (from scratch)",
            "runtime_ms": res_scratch["runtime_ms"],
            "num_itemsets": res_scratch["num_itemsets"],
            "num_rules": res_scratch["num_rules"],
        },
        "apriori": {
            "algorithm": "mlxtend Apriori",
            "runtime_ms": res_apriori["runtime_ms"],
            "num_itemsets": res_apriori["num_itemsets"],
            "num_rules": res_apriori["num_rules"],
        },
        "fpgrowth": {
            "algorithm": "mlxtend FP-Growth",
            "runtime_ms": res_fpgrowth["runtime_ms"],
            "num_itemsets": res_fpgrowth["num_itemsets"],
            "num_rules": res_fpgrowth["num_rules"],
        },
        "active_dataset_baskets": len(state.transactions),
    }


@app.get("/api/recommend")
def recommend_bundles(
    product: str = Query(..., description="ASIN or product title to get bundle recommendations for"),
    limit: int = Query(6, ge=1, le=20),
):
    """
    Return top recommended bundle rules involving the given product, ranked by lift.
    """
    if not state.latest_rules:
        # Run default mining if rules haven't been mined yet
        res = execute_mining(
            algorithm="scratch",
            transactions=state.transactions,
            one_hot_df=state.one_hot_df,
            min_support=0.02,
            min_confidence=0.30,
            min_lift=1.0,
            max_len=3,
            title_map=state.title_map,
        )
        state.latest_rules = res["rules"]

    target = product.strip().lower()

    # Find matching rules where target is in antecedent or consequent
    matching_rules = []
    for r in state.latest_rules:
        ant_ids = [x.lower() for x in r["antecedent"]]
        ant_titles = [x.lower() for x in r["antecedent_titles"]]
        con_ids = [x.lower() for x in r["consequent"]]
        con_titles = [x.lower() for x in r["consequent_titles"]]

        # Match either exact ID or substring in title
        hit_ant = any(target in x for x in ant_ids + ant_titles)
        hit_con = any(target in x for x in con_ids + con_titles)

        if hit_ant or hit_con:
            matching_rules.append(r)

    # Sort by lift descending
    matching_rules.sort(key=lambda x: x["lift"], reverse=True)

    return {
        "query": product,
        "recommendations_found": len(matching_rules),
        "bundles": matching_rules[:limit],
    }


@app.get("/api/sweep")
def parameter_sweep(
    min_rating: float = Query(4.0, ge=1.0, le=5.0),
):
    """
    Compute rule count grid over support {0.5%, 1%, 2%, 3%} and confidence {20%, 40%, 60%}
    for hyperparameter tuning analysis.
    Uses FP-Growth for high-speed computation.
    """
    if state.raw_df is None or state.one_hot_df.empty:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    state.update_rating_threshold(min_rating)

    supports = [0.005, 0.01, 0.02, 0.03]
    confidences = [0.20, 0.40, 0.60]

    matrix: List[List[int]] = []

    for s in supports:
        row: List[int] = []
        for c in confidences:
            res = run_mlxtend_fpgrowth(
                one_hot_df=state.one_hot_df,
                min_support=s,
                min_confidence=c,
                min_lift=1.0,
                max_len=3,
                title_map=state.title_map,
            )
            row.append(res["num_rules"])
        matrix.append(row)

    return {
        "supports": [f"{s*100:.1f}%" for s in supports],
        "confidences": [f"{c*100:.0f}%" for c in confidences],
        "supports_raw": supports,
        "confidences_raw": confidences,
        "grid": matrix,
        "recommended_params": {
            "min_support": 0.03,
            "min_confidence": 0.35,
            "min_lift": 1.2,
            "rationale": (
                "At 3% support and 35% confidence, we avoid combinatorial rule explosion "
                "while preserving strong, actionable grocery bundles with statistically significant lift."
            ),
        },
    }


@app.get("/api/manual-example")
def manual_example(
    rule_index: int = Query(0, ge=0),
):
    """
    Returns step-by-step arithmetic derivation for one association rule,
    showing raw transaction counts N, N(A), N(B), N(AB) and explicit formulas.
    """
    if not state.latest_rules:
        # Mine rules to have a rule pool
        res = execute_mining(
            algorithm="scratch",
            transactions=state.transactions,
            one_hot_df=state.one_hot_df,
            min_support=0.02,
            min_confidence=0.30,
            min_lift=1.0,
            max_len=3,
            title_map=state.title_map,
        )
        state.latest_rules = res["rules"]

    if not state.latest_rules:
        raise HTTPException(status_code=404, detail="No association rules available to demonstrate.")

    idx = min(rule_index, len(state.latest_rules) - 1)
    target_rule = state.latest_rules[idx]

    worked_example = format_manual_worked_example(
        antecedent=target_rule["antecedent"],
        consequent=target_rule["consequent"],
        antecedent_titles=target_rule["antecedent_titles"],
        consequent_titles=target_rule["consequent_titles"],
        count_a=target_rule["count_A"],
        count_b=target_rule["count_B"],
        count_ab=target_rule["count_AB"],
        total_transactions=target_rule["N"],
    )

    return worked_example


# -----------------------------------------------------------------------------
# Static File Mounting
# -----------------------------------------------------------------------------
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the single-page application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Basketry</h1><p>Frontend static files loading...</p>")
