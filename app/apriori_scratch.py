"""
Apriori Algorithm Implemented From Scratch.

Educational, clean, type-annotated implementation without mlxtend.
Implements:
1. Candidate 1-itemset generation (C1) and frequent 1-itemset filtering (L1).
2. Candidate k-itemset generation (Lk-1 join Lk-1).
3. Apriori pruning (downward closure: every subset of a frequent itemset must be frequent).
4. Support counting against transaction baskets.
5. Association rule derivation (antecedents and consequents).
6. Comprehensive metric evaluation: support, confidence, lift, leverage, conviction.
"""

import time
import math
import itertools
from typing import List, Dict, Set, Tuple, Any, Optional, FrozenSet
from app.metrics import (
    calculate_support,
    calculate_confidence,
    calculate_lift,
    calculate_leverage,
    calculate_conviction,
)


def _generate_candidates_k(
    prev_frequent_itemsets: List[Tuple[str, ...]],
    k: int,
    prev_frequent_set: Set[Tuple[str, ...]],
) -> List[Tuple[str, ...]]:
    """
    Candidate Generation Step (Lk-1 |><| Lk-1 join) + Apriori Pruning Step.

    Join Step:
        Two (k-1)-itemsets p and q are joined if they share their first (k-2) items
        in sorted order. Since p[k-2] < q[k-2], the joined candidate has length k.

    Prune Step:
        Verify that all (k-1)-subsets of the candidate itemset are present in
        prev_frequent_set. If ANY (k-1)-subset is not frequent, the candidate is
        immediately pruned without checking transaction data (downward closure property).
    """
    candidates: List[Tuple[str, ...]] = []
    n = len(prev_frequent_itemsets)

    for i in range(n):
        p = prev_frequent_itemsets[i]
        for j in range(i + 1, n):
            q = prev_frequent_itemsets[j]

            # Join condition: first k-2 elements match
            if p[: k - 2] == q[: k - 2]:
                # Combine: p + last element of q
                candidate = p + (q[k - 2],)

                # Apriori Prune condition:
                # Check all (k-1)-subsets of candidate
                all_subsets_frequent = True
                for subset in itertools.combinations(candidate, k - 1):
                    if subset not in prev_frequent_set:
                        all_subsets_frequent = False
                        break

                if all_subsets_frequent:
                    candidates.append(candidate)
            else:
                # Since list is sorted lexicographically, we can break early
                break

    return candidates


def apriori_from_scratch(
    transactions: List[List[str]],
    min_support: float = 0.03,
    min_confidence: float = 0.35,
    min_lift: float = 1.0,
    max_len: int = 3,
    title_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Mine frequent itemsets and association rules from a list of transactions.

    Parameters:
    -----------
    transactions : List[List[str]]
        List of baskets, where each basket is a list of product IDs / ASINs.
    min_support : float
        Minimum support threshold (0.0 to 1.0).
    min_confidence : float
        Minimum rule confidence threshold (0.0 to 1.0).
    min_lift : float
        Minimum rule lift threshold (default 1.0 for positive correlation).
    max_len : int
        Maximum itemset length to mine (default 3).
    title_map : Optional[Dict[str, str]]
        Dictionary mapping product IDs to human-readable titles.

    Returns:
    --------
    Dict containing:
        - algorithm: "scratch"
        - runtime_ms: execution time in milliseconds
        - num_itemsets: total count of frequent itemsets found
        - num_rules: total count of association rules found
        - itemsets: list of itemset dictionaries
        - rules: list of rule dictionaries sorted by lift descending
    """
    start_time = time.perf_counter()

    if title_map is None:
        title_map = {}

    N = len(transactions)
    if N == 0:
        return {
            "algorithm": "scratch",
            "runtime_ms": 0.0,
            "num_itemsets": 0,
            "num_rules": 0,
            "itemsets": [],
            "rules": [],
        }

    min_count = math.ceil(min_support * N)

    # Build inverted transaction index (item -> set of transaction indices)
    # Allows O(1) set-intersection counting at C-speed in Python
    item_tids: Dict[str, Set[int]] = {}
    for tx_idx, basket in enumerate(transactions):
        for item in basket:
            if item not in item_tids:
                item_tids[item] = set()
            item_tids[item].add(tx_idx)

    # Filter by min_count
    itemset_support_map: Dict[Tuple[str, ...], float] = {}
    itemset_count_map: Dict[Tuple[str, ...], int] = {}

    L1_tuples: List[Tuple[str, ...]] = []
    for item, tids in item_tids.items():
        count = len(tids)
        if count >= min_count:
            tup = (item,)
            L1_tuples.append(tup)
            itemset_support_map[tup] = count / N
            itemset_count_map[tup] = count

    # Sort lexicographically
    L1_tuples.sort()

    all_frequent_itemsets: List[Tuple[str, ...]] = list(L1_tuples)
    current_frequent: List[Tuple[str, ...]] = L1_tuples

    # ----------------------------------------------------
    # Step 2: Iterate to generate L2, L3, ... up to max_len
    # ----------------------------------------------------
    for k in range(2, max_len + 1):
        if not current_frequent:
            break

        prev_frequent_set = set(current_frequent)
        # Generate candidates of length k with join & pruning
        candidates = _generate_candidates_k(current_frequent, k, prev_frequent_set)

        if not candidates:
            break

        # Count support for each candidate using set intersection of transaction IDs
        candidate_counts: Dict[Tuple[str, ...], int] = {}
        for cand in candidates:
            common_tids = item_tids.get(cand[0], set())
            for item in cand[1:]:
                common_tids = common_tids & item_tids.get(item, set())
                if not common_tids:
                    break
            candidate_counts[cand] = len(common_tids)

        # Filter candidates meeting minimum support
        next_frequent: List[Tuple[str, ...]] = []
        for cand, count in candidate_counts.items():
            if count >= min_count:
                next_frequent.append(cand)
                itemset_support_map[cand] = count / N
                itemset_count_map[cand] = count

        next_frequent.sort()
        all_frequent_itemsets.extend(next_frequent)
        current_frequent = next_frequent

    # ----------------------------------------------------
    # Step 3: Rule Generation
    # For every frequent itemset I with |I| >= 2,
    # generate all non-empty antecedent subsets A, consequent B = I \ A
    # ----------------------------------------------------
    rules_list: List[Dict[str, Any]] = []

    for itemset in all_frequent_itemsets:
        k_items = len(itemset)
        if k_items < 2:
            continue

        itemset_supp = itemset_support_map[itemset]
        itemset_cnt = itemset_count_map[itemset]

        # Generate all proper non-empty subsets of itemset
        for r in range(1, k_items):
            for antecedent in itertools.combinations(itemset, r):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                supp_a = itemset_support_map.get(antecedent)
                supp_b = itemset_support_map.get(consequent)

                if supp_a is None or supp_b is None:
                    continue

                cnt_a = itemset_count_map[antecedent]
                cnt_b = itemset_count_map[consequent]

                # Confidence = count(A U B) / count(A) = support(A U B) / support(A)
                conf = itemset_supp / supp_a

                if conf < min_confidence:
                    continue

                # Lift = confidence / support(B)
                lift = conf / supp_b

                if lift < min_lift:
                    continue

                leverage = calculate_leverage(itemset_supp, supp_a, supp_b)
                conviction = calculate_conviction(supp_b, conf)

                rules_list.append({
                    "antecedent": list(antecedent),
                    "consequent": list(consequent),
                    "antecedent_titles": [title_map.get(item, item) for item in antecedent],
                    "consequent_titles": [title_map.get(item, item) for item in consequent],
                    "support": round(itemset_supp, 4),
                    "confidence": round(conf, 4),
                    "lift": round(lift, 3),
                    "leverage": round(leverage, 4),
                    "conviction": round(conviction, 3) if conviction != float("inf") else 999.0,
                    "count_A": cnt_a,
                    "count_B": cnt_b,
                    "count_AB": itemset_cnt,
                    "N": N,
                })

    # Sort rules by lift descending, then confidence descending
    rules_list.sort(key=lambda r: (r["lift"], r["confidence"]), reverse=True)

    # Format frequent itemsets for API response
    formatted_itemsets: List[Dict[str, Any]] = []
    for itemset in all_frequent_itemsets:
        supp = itemset_support_map[itemset]
        cnt = itemset_count_map[itemset]
        formatted_itemsets.append({
            "itemset": list(itemset),
            "titles": [title_map.get(item, item) for item in itemset],
            "support": round(supp, 4),
            "count": cnt,
            "length": len(itemset),
        })

    # Sort itemsets by support descending, then length descending
    formatted_itemsets.sort(key=lambda x: (x["support"], x["length"]), reverse=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "algorithm": "scratch",
        "runtime_ms": round(elapsed_ms, 2),
        "num_itemsets": len(formatted_itemsets),
        "num_rules": len(rules_list),
        "itemsets": formatted_itemsets,
        "rules": rules_list,
    }
