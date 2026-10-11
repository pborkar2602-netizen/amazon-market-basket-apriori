# Basketry: Market Basket Analysis on Amazon Product Reviews

## Project summary

This project studies market basket analysis on Amazon product reviews using association rule mining. Each reviewer is treated as a basket, and the products they reviewed are treated as the items in that basket. The repository includes data preparation, preprocessing, a custom Apriori implementation, comparisons with `mlxtend` Apriori and FP-Growth, a notebook with the analysis, a web interface, and a small test suite.

The project uses the Amazon Reviews 2023 dataset for the Grocery and Gourmet Food category. It filters for positive ratings and keeps only reviewers with more than one product in the basket. It then mines frequent itemsets and rule sets to identify product combinations that appear together more often than expected.

---

## 1. Data and problem setup

The analysis uses the Amazon Reviews 2023 dataset from the McAuley Lab at UC San Diego. The data is filtered to the Grocery and Gourmet Food category and then prepared into reviewer baskets.

A reviewer basket is represented as the set of products that reviewer has reviewed with a rating of at least 4.0. The project removes single-item baskets so that association rules have a meaningful co-occurrence signal.

---

## 2. Data preprocessing

The preprocessing pipeline performs the following steps:

- detect the relevant review columns and handle missing values
- remove duplicates by reviewer and product
- keep only ratings of 4.0 or higher
- drop reviewers with fewer than two products in the filtered basket
- convert the cleaned data to transaction lists and one-hot matrices for the mining algorithms

This keeps the basket data closer to products a user appears to like and reduces noise from low-quality or single-item interactions.

---

## 3. Mining methods

The project implements an Apriori algorithm from scratch in `app/apriori_scratch.py`. It generates candidate itemsets, applies the downward-closure rule, counts support, and builds rules from frequent itemsets.

The project also compares this implementation with `mlxtend` Apriori and `mlxtend` FP-Growth in `app/algorithms.py`. The comparison checks whether the discovered itemsets and rules match while measuring runtime differences.

---

## 4. Metrics

The analysis uses standard association-rule metrics:

- support
- confidence
- lift
- leverage
- conviction

These measures are implemented in `app/metrics.py` and are used to rank and interpret the generated rules.

---

## 5. Results

The demo dataset includes 1,809 valid baskets after filtering. The mining parameters are set to a minimum support of 0.03, minimum confidence of 0.35, minimum lift of 1.2, and a maximum itemset length of 3.

Under these settings, the scratch Apriori, `mlxtend` Apriori, and `mlxtend` FP-Growth produce the same frequent itemsets and rules. The rules indicate product combinations that recur together more often than expected by chance.

---

## 6. Limitations

Reviews are not purchases. A reviewer may have bought items at different times or from different sellers, and a review does not necessarily correspond to a single shopping basket. The results should therefore be read as evidence of co-review or co-interest patterns, not as direct purchase data.

---

## 7. Repository structure

- `app/` contains the preprocessing, Apriori, metrics, and API code.
- `data/` contains the data preparation script and sample CSV files.
- `docs/` stores screenshots referenced in the README.
- `notebook/analysis.ipynb` contains the analysis notebook.
- `reports/REPORT.md` is the project report.
- `tests/` contains the unit and integration tests.

---

## 8. Citation

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
