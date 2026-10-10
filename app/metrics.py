"""
Association Rule Mining Metrics module.

Defines mathematical metrics:
- Support
- Confidence
- Lift
- Leverage
- Conviction

Also provides a worked manual step-by-step arithmetic breakdown for demonstration and grading.
"""

from typing import Dict, Any, List, Optional, Union
import math


def calculate_support(count_itemset: int, total_transactions: int) -> float:
    """
    Support(I) = count(I) / N
    Fraction of all transactions that contain the itemset I.
    """
    if total_transactions <= 0:
        return 0.0
    return count_itemset / total_transactions


def calculate_confidence(count_both: int, count_antecedent: int) -> float:
    """
    Confidence(A -> B) = count(A U B) / count(A) = P(B | A)
    Probability that a basket contains B given that it already contains A.
    """
    if count_antecedent <= 0:
        return 0.0
    return count_both / count_antecedent


def calculate_lift(support_both: float, support_antecedent: float, support_consequent: float) -> float:
    """
    Lift(A -> B) = Confidence(A -> B) / Support(B) = Support(A U B) / (Support(A) * Support(B))
    Ratio of observed joint support to expected support under independence.
    - Lift = 1: A and B are completely independent.
    - Lift > 1: Positive affinity; buying A makes buying B significantly more likely.
    - Lift < 1: Negative affinity / substitutes; buying A makes buying B less likely.
    """
    denom = support_antecedent * support_consequent
    if denom <= 0:
        return 0.0
    return support_both / denom


def calculate_leverage(support_both: float, support_antecedent: float, support_consequent: float) -> float:
    """
    Leverage(A -> B) = Support(A U B) - (Support(A) * Support(B))
    Difference between the observed frequency of A and B appearing together
    and the frequency expected if A and B were statistically independent.
    Bounded in [-0.25, 0.25].
    """
    return support_both - (support_antecedent * support_consequent)


def calculate_conviction(support_consequent: float, confidence_both: float) -> float:
    """
    Conviction(A -> B) = (1 - Support(B)) / (1 - Confidence(A -> B))
    Ratio of expected frequency that A occurs without B if independent,
    to observed frequency of incorrect predictions.
    Value of infinity represents a perfect rule (Confidence = 1).
    """
    if math.isclose(confidence_both, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return float("inf")
    denom = 1.0 - confidence_both
    if denom <= 0:
        return float("inf")
    return (1.0 - support_consequent) / denom


def format_manual_worked_example(
    antecedent: List[str],
    consequent: List[str],
    antecedent_titles: List[str],
    consequent_titles: List[str],
    count_a: int,
    count_b: int,
    count_ab: int,
    total_transactions: int,
) -> Dict[str, Any]:
    """
    Produces a complete, step-by-step arithmetic derivation of support, confidence,
    lift, leverage, and conviction for a single association rule.
    """
    N = total_transactions
    supp_a = calculate_support(count_a, N)
    supp_b = calculate_support(count_b, N)
    supp_ab = calculate_support(count_ab, N)
    conf = calculate_confidence(count_ab, count_a)
    lift_val = calculate_lift(supp_ab, supp_a, supp_b)
    leverage_val = calculate_leverage(supp_ab, supp_a, supp_b)
    conviction_val = calculate_conviction(supp_b, conf)

    a_label = " + ".join(antecedent_titles)
    b_label = " + ".join(consequent_titles)
    rule_str = f"[{a_label}] => [{b_label}]"

    steps = [
        {
            "metric": "Total Dataset Size (N)",
            "formula": "N = total qualifying multi-item baskets",
            "calculation": f"N = {N:,} baskets",
            "result": f"{N:,}",
            "interpretation": "Total number of reviewer baskets with >= 2 positive reviews.",
        },
        {
            "metric": "Raw Occurrence Counts",
            "formula": "count(A), count(B), count(A U B)",
            "calculation": f"count(A) = {count_a:,} | count(B) = {count_b:,} | count(A U B) = {count_ab:,}",
            "result": f"{count_ab:,} joint baskets",
            "interpretation": f"{count_a} reviewers reviewed A, {count_b} reviewed B, and {count_ab} reviewed both.",
        },
        {
            "metric": "Support(A)",
            "formula": "Support(A) = count(A) / N",
            "calculation": f"{count_a} / {N} = {supp_a:.4f}",
            "result": f"{supp_a * 100:.2f}%",
            "interpretation": f"{supp_a * 100:.2f}% of all reviewer baskets contain the antecedent.",
        },
        {
            "metric": "Support(B)",
            "formula": "Support(B) = count(B) / N",
            "calculation": f"{count_b} / {N} = {supp_b:.4f}",
            "result": f"{supp_b * 100:.2f}%",
            "interpretation": f"{supp_b * 100:.2f}% of all reviewer baskets contain the consequent.",
        },
        {
            "metric": "Rule Support (A U B)",
            "formula": "Support(A => B) = count(A U B) / N",
            "calculation": f"{count_ab} / {N} = {supp_ab:.4f}",
            "result": f"{supp_ab * 100:.2f}%",
            "interpretation": f"Both products appear together in {supp_ab * 100:.2f}% of all baskets.",
        },
        {
            "metric": "Confidence (A => B)",
            "formula": "Confidence(A => B) = count(A U B) / count(A)",
            "calculation": f"{count_ab} / {count_a} = {conf:.4f}",
            "result": f"{conf * 100:.2f}%",
            "interpretation": f"Of reviewers who reviewed A, {conf * 100:.2f}% also reviewed B.",
        },
        {
            "metric": "Lift (A => B)",
            "formula": "Lift = Confidence(A => B) / Support(B) = Support(A U B) / (Support(A) * Support(B))",
            "calculation": f"{conf:.4f} / {supp_b:.4f} = {lift_val:.2f}x",
            "result": f"{lift_val:.2f}x",
            "interpretation": (
                f"Reviewers who reviewed A are {lift_val:.2f} times more likely to review B "
                f"than a random reviewer. (Lift > 1 proves a strong bundle synergy)."
            ),
        },
        {
            "metric": "Leverage (A => B)",
            "formula": "Leverage = Support(A U B) - (Support(A) * Support(B))",
            "calculation": f"{supp_ab:.4f} - ({supp_a:.4f} * {supp_b:.4f}) = {leverage_val:.4f}",
            "result": f"{leverage_val:.4f}",
            "interpretation": (
                f"The co-occurrence exceeds what would be expected under random independence by "
                f"{leverage_val * 100:.2f}% of the total transaction volume."
            ),
        },
        {
            "metric": "Conviction (A => B)",
            "formula": "Conviction = (1 - Support(B)) / (1 - Confidence(A => B))",
            "calculation": (
                f"(1 - {supp_b:.4f}) / (1 - {conf:.4f}) = "
                f"{1.0 - supp_b:.4f} / {1.0 - conf:.4f} = {conviction_val:.2f}"
                if conviction_val != float("inf")
                else "Infinity (rule holds with 100% confidence)"
            ),
            "result": f"{conviction_val:.2f}" if conviction_val != float("inf") else "inf",
            "interpretation": (
                f"Measures the degree of directional implication. "
                f"A conviction of {conviction_val:.2f} indicates that the rule would be incorrect "
                f"{conviction_val:.2f} times as often if the items were purely independent."
            ),
        },
    ]

    return {
        "rule_display": rule_str,
        "antecedent": antecedent,
        "consequent": consequent,
        "antecedent_titles": antecedent_titles,
        "consequent_titles": consequent_titles,
        "raw_counts": {
            "N": N,
            "count_A": count_a,
            "count_B": count_b,
            "count_AB": count_ab,
        },
        "metrics": {
            "support": round(supp_ab, 4),
            "confidence": round(conf, 4),
            "lift": round(lift_val, 3),
            "leverage": round(leverage_val, 4),
            "conviction": round(conviction_val, 3) if conviction_val != float("inf") else "inf",
        },
        "steps": steps,
    }
