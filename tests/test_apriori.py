"""
Unit test suite for Basketry association rule mining.

Verifies:
1. Metric calculation formulas (support, confidence, lift, leverage, conviction).
2. Equivalence between scratch Apriori and mlxtend Apriori on demo_small.csv.
3. Preprocessing functionality (null handling, rating filter, dropping single items).
"""

import os
import math
import pytest
import pandas as pd
import numpy as np

from app.metrics import (
    calculate_support,
    calculate_confidence,
    calculate_lift,
    calculate_leverage,
    calculate_conviction,
    format_manual_worked_example,
)
from app.preprocess import detect_columns, clean_and_prepare_baskets, summarize_dataset
from app.apriori_scratch import apriori_from_scratch
from app.algorithms import run_mlxtend_apriori, run_mlxtend_fpgrowth


def test_metric_formulas():
    """Verify standard metric mathematical formulas."""
    N = 1000
    cnt_a = 200
    cnt_b = 300
    cnt_ab = 100

    supp_a = calculate_support(cnt_a, N)
    supp_b = calculate_support(cnt_b, N)
    supp_ab = calculate_support(cnt_ab, N)

    assert math.isclose(supp_a, 0.20, abs_tol=1e-5)
    assert math.isclose(supp_b, 0.30, abs_tol=1e-5)
    assert math.isclose(supp_ab, 0.10, abs_tol=1e-5)

    conf = calculate_confidence(cnt_ab, cnt_a)
    assert math.isclose(conf, 0.50, abs_tol=1e-5)

    lift_val = calculate_lift(supp_ab, supp_a, supp_b)
    # Lift = 0.50 / 0.30 = 1.6666...
    assert math.isclose(lift_val, 1.666667, abs_tol=1e-4)

    leverage_val = calculate_leverage(supp_ab, supp_a, supp_b)
    # Leverage = 0.10 - (0.20 * 0.30) = 0.10 - 0.06 = 0.04
    assert math.isclose(leverage_val, 0.04, abs_tol=1e-5)

    conviction_val = calculate_conviction(supp_b, conf)
    # Conviction = (1 - 0.30) / (1 - 0.50) = 0.70 / 0.50 = 1.40
    assert math.isclose(conviction_val, 1.40, abs_tol=1e-5)


def test_manual_worked_example_formatting():
    """Verify manual worked example produces structured output with all counts and steps."""
    res = format_manual_worked_example(
        antecedent=["B001"],
        consequent=["B002"],
        antecedent_titles=["Pasta"],
        consequent_titles=["Sauce"],
        count_a=150,
        count_b=200,
        count_ab=90,
        total_transactions=1000,
    )

    assert "rule_display" in res
    assert "[Pasta] => [Sauce]" in res["rule_display"]
    assert res["raw_counts"]["count_AB"] == 90
    assert res["metrics"]["support"] == 0.09
    assert res["metrics"]["confidence"] == 0.60
    assert len(res["steps"]) >= 8


def test_preprocessing_pipeline():
    """Test preprocessing and basket grouping with single-item basket dropping."""
    raw_data = pd.DataFrame([
        {"reviewerID": "U1", "asin": "P1", "rating": 5.0, "title": "Item 1"},
        {"reviewerID": "U1", "asin": "P2", "rating": 5.0, "title": "Item 2"},
        {"reviewerID": "U2", "asin": "P1", "rating": 5.0, "title": "Item 1"},  # Single item basket!
        {"reviewerID": "U3", "asin": "P2", "rating": 2.0, "title": "Item 2"},  # Low rating!
        {"reviewerID": "U3", "asin": "P3", "rating": 5.0, "title": "Item 3"},
        {"reviewerID": "U4", "asin": "P1", "rating": 4.0, "title": "Item 1"},
        {"reviewerID": "U4", "asin": "P2", "rating": 4.0, "title": "Item 2"},
        {"reviewerID": "U4", "asin": "P3", "rating": 4.0, "title": "Item 3"},
    ])

    transactions, one_hot_df, title_map, meta = clean_and_prepare_baskets(raw_data, min_rating=4.0)

    # U1 has P1, P2 (size 2 -> kept)
    # U2 has only P1 (size 1 -> dropped)
    # U3 has P3 (P2 was 2.0, dropped, leaving only P3 -> size 1 -> dropped)
    # U4 has P1, P2, P3 (size 3 -> kept)
    assert len(transactions) == 2
    assert meta["qualifying_baskets"] == 2
    assert meta["single_item_baskets_dropped"] == 2
    assert "P1" in title_map and title_map["P1"] == "Item 1"


def test_scratch_vs_mlxtend_equivalence():
    """
    Core verification required by project specification:
    Verify scratch Apriori results match mlxtend on demo_small.csv.
    """
    demo_csv = os.path.join(os.path.dirname(__file__), "..", "data", "demo_small.csv")
    assert os.path.exists(demo_csv), f"Demo data file {demo_csv} not found."

    df = pd.read_csv(demo_csv)
    transactions, one_hot_df, title_map, meta = clean_and_prepare_baskets(df, min_rating=4.0)

    min_support = 0.03
    min_confidence = 0.30
    min_lift = 1.2
    max_len = 3

    scratch_res = apriori_from_scratch(
        transactions=transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=title_map,
    )

    mlxtend_res = run_mlxtend_apriori(
        one_hot_df=one_hot_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=title_map,
    )

    fpgrowth_res = run_mlxtend_fpgrowth(
        one_hot_df=one_hot_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
        title_map=title_map,
    )

    # Compare frequent itemsets count
    scratch_itemsets = {tuple(sorted(x["itemset"])) for x in scratch_res["itemsets"]}
    mlxtend_itemsets = {tuple(sorted(x["itemset"])) for x in mlxtend_res["itemsets"]}
    fpgrowth_itemsets = {tuple(sorted(x["itemset"])) for x in fpgrowth_res["itemsets"]}

    assert scratch_itemsets == mlxtend_itemsets, (
        f"Itemsets mismatch: scratch found {len(scratch_itemsets)}, mlxtend found {len(mlxtend_itemsets)}"
    )
    assert mlxtend_itemsets == fpgrowth_itemsets, (
        f"Itemsets mismatch: mlxtend found {len(mlxtend_itemsets)}, fpgrowth found {len(fpgrowth_itemsets)}"
    )

    # Compare rules
    scratch_rules = {
        (tuple(sorted(r["antecedent"])), tuple(sorted(r["consequent"])))
        for r in scratch_res["rules"]
    }
    mlxtend_rules = {
        (tuple(sorted(r["antecedent"])), tuple(sorted(r["consequent"])))
        for r in mlxtend_res["rules"]
    }

    assert scratch_rules == mlxtend_rules, (
        f"Rules mismatch: scratch found {len(scratch_rules)}, mlxtend found {len(mlxtend_rules)}"
    )
    assert len(scratch_rules) > 0, "Expected to find frequent association rules in demo data."
