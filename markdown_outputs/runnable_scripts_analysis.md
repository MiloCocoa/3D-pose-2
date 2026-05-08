# Repository Runnable Python Scripts Analysis

This document outlines the purpose of each runnable Python script in the repository.

| Script | Purpose |
| :--- | :--- |
| `analyze_labels.py` | Performs exploratory data analysis to count the occurrences of `Descent` and `Ascent` labels in the provided JSON dataset. Used to assess data imbalance. |
| `calibrate_rules.py` | Calculates and evaluates the performance (precision, recall, f1) of the rule-based geometric engines (e.g., knee valgus, depth) against a labeled dataset to optimize threshold values. |
| `debug_data.py` | Performs sanity checks on the JSON pose data, scanning for `NaN`, `Inf`, or extreme coordinate values that could indicate sensor/tracking errors. |
| `evaluate_v2.py` | Evaluates the performance of the trained ST-GCN model (`multi_label_gcn_v2.pth`) on a test set, outputting precision, recall, and f1 scores. |
| `find_optimal_thresholds.py` | Runs a systematic sweep of threshold values across the geometric rules to determine the optimal cutoff points for maximizing classification accuracy. |
| `split_data.py` | Splits raw 3D pose data into separate `train` and `test` directories while ensuring no data leakage, facilitating robust model training and validation. |
| `trim_data.py` | Cleans or trims pose sequence data to standardize sequence lengths or remove invalid frame segments, ensuring compatibility with the data loader. |
