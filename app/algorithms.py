"""
Algorithm wrappers for mlxtend Apriori and FP-Growth.

Standardizes results across:
1. Apriori from scratch (app.apriori_scratch)
2. mlxtend Apriori
3. mlxtend FP-Growth

All three return identical dictionary shapes with matching metrics and titles.
"""

import time
import math
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth
from mlxtend.frequent_patterns import association_rules as mlx_assoc_rules

from app.apriori_scratch import apriori_from_scratch
from app.metrics import calculate_leverage, calculate_conviction


def _format_mlxtend_results(
    freq_df: pd.DataFrame,
    rules_df: Optional[pd.DataFrame],
    algorithm_name: str,
    elapsed_ms: float,
    N: int,
    title_map: Dict[str, str],
    min_lift: float,
) -> Dict[str, Any]:
    """
    Format mlxtend DataFrames into the common schema used by Basketry.
    """
    # Format frequent itemsets
    formatted_itemsets: List[Dict[str, Any]] = []
    if not freq_df.empty:
        # Sort by support descending
        freq_sorted = freq_df.sort_values(by="support", ascending=False)
        for _, row in freq_sorted.iterrows():
            items = sorted(list(row["itemsets"]))
            supp = float(row["support"])
            formatted_itemsets.append({
                "itemset": items,
                "titles": [title_map.get(i, i) for i in items],
                "support": round(supp, 4),
                "count": int(round(supp * N)),
                "length": len(items),
            })

    # Format association rules
    formatted_rules: List[Dict[str, Any]] = []
    if rules_df is not None and not rules_df.empty:
        # Filter by min_lift
        filtered_rules = rules_df[rules_df["lift"] >= min_lift].copy()
        # Sort by lift descending, then confidence descending
        filtered_rules = filtered_rules.sort_values(by=["lift", "confidence"], ascending=[False, False])

        for _, row in filtered_rules.iterrows():
            ant = sorted(list(row["antecedents"]))
            con = sorted(list(row["consequents"]))
            supp = float(row["support"])
            conf = float(row["confidence"])
            lift_val = float(row["lift"])
            leverage_val = float(row.get("leverage", calculate_leverage(supp, float(row["antecedent support"]), float(row["consequent support"]))))
            conviction_val = float(row.get("conviction", 999.0))
            if math.isnan(conviction_val) or math.isinf(conviction_val):
                conviction_val = 999.0

            cnt_a = int(round(float(row["antecedent support"]) * N))
            cnt_b = int(round(float(row["consequent support"]) * N))
            cnt_ab = int(round(supp * N))

            formatted_rules.append({
                "antecedent": ant,
                "consequent": con,
                "antecedent_titles": [title_map.get(i, i) for i in ant],
                "consequent_titles": [title_map.get(i, i) for i in con],
                "support": round(supp, 4),
                "confidence": round(conf, 4),
                "lift": round(lift_val, 3),
                "leverage": round(leverage_val, 4),
                "conviction": round(conviction_val, 3),
                "count_A": cnt_a,
                "count_B": cnt_b,
                "count_AB": cnt_ab,
                "N": N,
            })

    return {
        "algorithm": algorithm_name,
        "runtime_ms": round(elapsed_ms, 2),
        "num_itemsets": len(formatted_itemsets),
        "num_rules": len(formatted_rules),
        "itemsets": formatted_itemsets,
        "rules": formatted_rules,
    }


def run_mlxtend_apriori(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.35,
    min_lift: float = 1.0,
    max_len: int = 3,
    title_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run mlxtend Apriori on binary one-hot encoded transaction DataFrame.
    """
    if title_map is None:
        title_map = {}

    N = len(one_hot_df)
    if N == 0 or one_hot_df.empty:
        return {
            "algorithm": "apriori",
            "runtime_ms": 0.0,
            "num_itemsets": 0,
            "num_rules": 0,
            "itemsets": [],
            "rules": [],
        }

    start_time = time.perf_counter()

    freq_df = mlx_apriori(
        one_hot_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    rules_df = None
    if not freq_df.empty:
        # mlxtend association_rules requires at least one itemset of length >= 2
        has_multis = any(len(items) >= 2 for items in freq_df["itemsets"])
        if has_multis:
            try:
                rules_df = mlx_assoc_rules(
                    freq_df,
                    metric="confidence",
                    min_threshold=min_confidence,
                )
            except ValueError:
                rules_df = None

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return _format_mlxtend_results(
        freq_df=freq_df,
        rules_df=rules_df,
        algorithm_name="apriori",
        elapsed_ms=elapsed_ms,
        N=N,
        title_map=title_map,
        min_lift=min_lift,
    )


def run_mlxtend_fpgrowth(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.35,
    min_lift: float = 1.0,
    max_len: int = 3,
    title_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run mlxtend FP-Growth on binary one-hot encoded transaction DataFrame.
    """
    if title_map is None:
        title_map = {}

    N = len(one_hot_df)
    if N == 0 or one_hot_df.empty:
        return {
            "algorithm": "fpgrowth",
            "runtime_ms": 0.0,
            "num_itemsets": 0,
            "num_rules": 0,
            "itemsets": [],
            "rules": [],
        }

    start_time = time.perf_counter()

    freq_df = mlx_fpgrowth(
        one_hot_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    rules_df = None
    if not freq_df.empty:
        has_multis = any(len(items) >= 2 for items in freq_df["itemsets"])
        if has_multis:
            try:
                rules_df = mlx_assoc_rules(
                    freq_df,
                    metric="confidence",
                    min_threshold=min_confidence,
                )
            except ValueError:
                rules_df = None

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return _format_mlxtend_results(
        freq_df=freq_df,
        rules_df=rules_df,
        algorithm_name="fpgrowth",
        elapsed_ms=elapsed_ms,
        N=N,
        title_map=title_map,
        min_lift=min_lift,
    )


def execute_mining(
    algorithm: str,
    transactions: List[List[str]],
    one_hot_df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.35,
    min_lift: float = 1.0,
    max_len: int = 3,
    title_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint to run any of the three algorithms:
    - 'scratch': Custom Apriori from scratch
    - 'apriori': mlxtend Apriori
    - 'fpgrowth': mlxtend FP-Growth
    """
    algo = algorithm.lower().strip()
    if algo == "scratch":
        return apriori_from_scratch(
            transactions=transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            title_map=title_map,
        )
    elif algo == "apriori":
        return run_mlxtend_apriori(
            one_hot_df=one_hot_df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            title_map=title_map,
        )
    elif algo == "fpgrowth":
        return run_mlxtend_fpgrowth(
            one_hot_df=one_hot_df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            title_map=title_map,
        )
    else:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Choose from 'scratch', 'apriori', 'fpgrowth'.")
