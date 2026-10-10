"""
Preprocessing module for Basketry: Market Basket Analysis.

Handles:
- Column auto-detection (reviewer, product, rating, title)
- Data cleaning (null handling, deduplication)
- Rating thresholding (e.g. rating >= 4.0 for positive affinity)
- Dropping single-item baskets (cannot form co-occurrence pairs)
- Basket representations: transaction lists and one-hot binary matrices
"""

import re
from typing import Tuple, List, Dict, Set, Any, Optional
import pandas as pd
import numpy as np


# Candidate regex patterns for column auto-detection
REVIEWER_COLS = [r"reviewer.*id", r"user.*id", r"customer.*id", r"reviewer", r"user", r"customer", r"author"]
PRODUCT_COLS = [r"asin", r"product.*id", r"item.*id", r"product", r"item", r"sku"]
RATING_COLS = [r"rating", r"overall", r"score", r"stars", r"stars.*rating"]
TITLE_COLS = [r"title", r"product.*name", r"item.*name", r"name", r"description"]


def detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Auto-detect reviewer, product, rating, and title columns from a list of dataframe headers.
    """
    detected: Dict[str, Optional[str]] = {
        "reviewer": None,
        "product": None,
        "rating": None,
        "title": None,
    }

    cols_lower = [c.strip().lower() for c in columns]

    def find_match(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            for idx, c in enumerate(cols_lower):
                if re.fullmatch(pat, c) or re.search(pat, c):
                    return columns[idx]
        return None

    detected["reviewer"] = find_match(REVIEWER_COLS)
    detected["product"] = find_match(PRODUCT_COLS)
    detected["rating"] = find_match(RATING_COLS)
    detected["title"] = find_match(TITLE_COLS)

    return detected


def summarize_dataset(df: pd.DataFrame, col_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Compute summary statistics of the raw uploaded or loaded dataset:
    total rows, unique reviewers, unique products, null counts, duplicate count,
    and rating distribution.
    """
    rev_col = col_map.get("reviewer")
    prod_col = col_map.get("product")
    rat_col = col_map.get("rating")

    total_rows = int(len(df))
    num_reviewers = int(df[rev_col].nunique()) if rev_col and rev_col in df else 0
    num_products = int(df[prod_col].nunique()) if prod_col and prod_col in df else 0

    null_counts = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
    total_nulls = int(df.isnull().sum().sum())

    # Check duplicates on reviewer + product
    if rev_col and prod_col and rev_col in df and prod_col in df:
        duplicate_count = int(df.duplicated(subset=[rev_col, prod_col]).sum())
    else:
        duplicate_count = int(df.duplicated().sum())

    # Rating distribution
    rating_dist: Dict[str, int] = {}
    if rat_col and rat_col in df:
        # Round ratings to nearest integer between 1 and 5
        ratings_clean = pd.to_numeric(df[rat_col], errors="coerce").dropna().round().astype(int)
        val_counts = ratings_clean.value_counts().to_dict()
        for star in [1, 2, 3, 4, 5]:
            rating_dist[f"{star}_star"] = int(val_counts.get(star, 0))
    else:
        # Default all 5 stars if no rating column
        rating_dist = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": total_rows}

    return {
        "rows": total_rows,
        "reviewers": num_reviewers,
        "products": num_products,
        "null_count": total_nulls,
        "nulls_by_column": null_counts,
        "duplicate_count": duplicate_count,
        "rating_distribution": rating_dist,
        "detected_columns": col_map,
    }


def clean_and_prepare_baskets(
    df: pd.DataFrame,
    min_rating: float = 4.0,
    col_map: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[List[str]], pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """
    Preprocess the raw reviews dataset into transaction baskets:
    1. Filter missing reviewer or product IDs.
    2. Drop duplicate reviewer-product interactions (keeping highest rating).
    3. Filter by minimum rating threshold (e.g. >= 4.0). Rationale:
       reviews are positive signals of customer endorsement, not mere purchases.
    4. Group products by reviewer into baskets.
    5. Drop single-item baskets (size < 2). Rationale: single-item baskets
       cannot contribute to item pairs or association rules; including them
       artificially inflates transaction count N and deflates support.
    6. Construct one-hot binary DataFrame for mlxtend algorithms.

    Returns:
    - transactions: List of product ID lists per qualifying reviewer
    - one_hot_df: Binary boolean DataFrame (rows=reviewers, cols=products)
    - title_map: Mapping from product ID (ASIN) to readable title
    - metadata: Dict with filtering statistics (raw vs kept reviewers/baskets)
    """
    if col_map is None:
        col_map = detect_columns(list(df.columns))

    rev_col = col_map.get("reviewer")
    prod_col = col_map.get("product")
    rat_col = col_map.get("rating")
    title_col = col_map.get("title")

    if not rev_col or not prod_col:
        raise ValueError(
            "Could not identify reviewer and product columns. "
            "Please ensure your CSV has 'reviewerID' (or 'user') and 'asin' (or 'product')."
        )

    # Make working copy of relevant columns
    cols_to_use = [rev_col, prod_col]
    if rat_col and rat_col in df:
        cols_to_use.append(rat_col)
    if title_col and title_col in df:
        cols_to_use.append(title_col)

    work_df = df[cols_to_use].copy()

    # Drop nulls in primary identifiers
    work_df = work_df.dropna(subset=[rev_col, prod_col])

    # Build title map
    title_map: Dict[str, str] = {}
    if title_col and title_col in work_df:
        # Get most frequent non-null title per product
        title_df = work_df.dropna(subset=[prod_col, title_col])
        for prod_id, group in title_df.groupby(prod_col):
            title_map[str(prod_id)] = str(group[title_col].iloc[0])

    # Convert product and reviewer IDs to string
    work_df[rev_col] = work_df[rev_col].astype(str)
    work_df[prod_col] = work_df[prod_col].astype(str)

    # Populate any missing product titles with product ID
    for prod_id in work_df[prod_col].unique():
        if prod_id not in title_map:
            title_map[prod_id] = prod_id

    # Handle ratings
    if rat_col and rat_col in work_df:
        work_df[rat_col] = pd.to_numeric(work_df[rat_col], errors="coerce").fillna(5.0)
        # Deduplicate reviewer-product pairs by taking the max rating
        work_df = work_df.sort_values(by=rat_col, ascending=False).drop_duplicates(subset=[rev_col, prod_col])
        # Filter by min_rating
        filtered_df = work_df[work_df[rat_col] >= min_rating]
    else:
        work_df = work_df.drop_duplicates(subset=[rev_col, prod_col])
        filtered_df = work_df

    total_reviewers_before = int(work_df[rev_col].nunique())

    # Group into baskets
    grouped = filtered_df.groupby(rev_col)[prod_col].apply(lambda s: sorted(list(set(s))))

    # Drop single-item baskets (basket size >= 2)
    multi_item_baskets = grouped[grouped.apply(len) >= 2]
    transactions: List[List[str]] = multi_item_baskets.tolist()

    qualifying_reviewers = len(transactions)
    unique_items_in_baskets = len(set(item for basket in transactions for item in basket))

    # Build one-hot encoded matrix for mlxtend algorithms
    if transactions:
        # Gather all unique items
        unique_items = sorted(list(set(item for basket in transactions for item in basket)))
        item_to_idx = {item: i for i, item in enumerate(unique_items)}
        n_tx = len(transactions)
        n_items = len(unique_items)

        # Build boolean array
        one_hot_arr = np.zeros((n_tx, n_items), dtype=bool)
        for row_idx, basket in enumerate(transactions):
            for item in basket:
                one_hot_arr[row_idx, item_to_idx[item]] = True

        one_hot_df = pd.DataFrame(one_hot_arr, columns=unique_items, index=multi_item_baskets.index)
    else:
        one_hot_df = pd.DataFrame()

    meta = {
        "raw_rows": len(df),
        "raw_reviewers": total_reviewers_before,
        "min_rating": min_rating,
        "qualifying_baskets": qualifying_reviewers,
        "unique_products_in_baskets": unique_items_in_baskets,
        "single_item_baskets_dropped": int((grouped.apply(len) < 2).sum()),
        "average_basket_size": float(np.mean([len(b) for b in transactions])) if transactions else 0.0,
    }

    return transactions, one_hot_df, title_map, meta
