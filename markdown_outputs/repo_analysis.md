# Repository Analysis Report

### 🎯 Objective
The project aims to build a 3D Barbell Squat Form Correction system utilizing 3D pose data from MediaPipe to provide real-time and offline feedback on squat form. 

The system relies on a **Dual-Head Architecture**:
1. **Rule-Based Head (Geometric/Safety):** Tracks deterministic labels using explicit geometric thresholds and synthesizes "Virtual Nodes" to improve accuracy.
2. **AI Head (ST-GCN - Postural/Mechanics):** A Spatio-Temporal Graph Convolutional Network that evaluates complex, time-dependent labels.

A React-based frontend is used to visualize a 3D skeleton with a dynamic heat map that highlights form mistakes.

### 📈 Current Progress
*   **Dual-Head Setup:** The core architecture separating rule-based and AI-based logic has been successfully integrated.
*   **Visual Polish:** The 3D React skeleton has been enhanced with proportional severity rendering and a colored gradient to clearly identify mistakes based on severity scores.
*   **Rule Calibration:** Rule-based thresholds were updated to a "Lenient Baseline" to reduce false positives and skeleton flickering caused by MediaPipe jitter.
*   **Data Integrity:** The dataset was manually split into training, testing, and validation directories, and data loaders have been updated to strictly prevent data leakage.

### ⚠️ Known Issues
*   **Class Imbalance:** There is a severe class imbalance within the training set, specifically for the `Descent` and `Ascent` labels, heavily favoring "Pass" sequences over "Fail" sequences.
*   **Model Blindness:** Due to this imbalance, the ST-GCN model predicts "Pass" for everything, resulting in a failure to identify actual mistakes for those labels despite high overall accuracy.

### 🚀 Next Steps
*   **Data Augmentation:** Synthetically increase the representation of minority "Fail" cases in the training dataset to provide the model with sufficient examples to learn from.
*   **Loss Re-Weighting:** Adjust the loss function to penalize the model more heavily for missing the minority class, forcing it to prioritize learning the failure patterns.
*   **Hyperparameter Tuning:** Fine-tune model parameters to mitigate overfitting to the newly augmented data.
*   **Retrain & Evaluate:** Retrain the model and evaluate its performance to ensure it can adequately detect minority classes.
*   **Rule Validation:** After stabilizing the AI Head, empirically test the new rule thresholds on the validation set to ensure accuracy.
