import os
import json
import numpy as np
import torch
from src.inference import InferenceEngine
from src.config import DATA_DIR, LABELS, MODEL_SAVE_DIR, MODEL_NAME
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_class_distribution(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    counts = np.zeros(10)
    total = 0
    for fname in files:
        with open(os.path.join(directory, fname), 'r') as f:
            data = json.load(f)
        raw_labels = data.get('metadata', {}).get('label')
        if raw_labels and len(raw_labels) == 10:
            total += 1
            for i in range(10):
                if raw_labels[i] == "False": # Fail
                    counts[i] += 1
    return counts, total

def evaluate_full_system():
    # Folders
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found.")
        return

    # Distribution Info
    train_counts, train_total = get_class_distribution(train_dir)
    test_counts, test_total = get_class_distribution(test_dir)

    # Initialize Inference Engine
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    engine = InferenceEngine(model_path)

    print("\n" + "="*85)
    print("DATASET CLASS DISTRIBUTION (FAIL / TOTAL)")
    print("-" * 85)
    print(f"{'Label':<15} | {'Train Fail':<10} | {'Train Total':<11} | {'Test Fail':<10} | {'Test Total':<10}")
    print("-" * 85)
    for i, name in enumerate(LABELS):
        print(f"{name:<15} | {int(train_counts[i]):<10} | {train_total:<11} | {int(test_counts[i]):<10} | {test_total:<10}")
    print("="*85 + "\n")

    all_y_true = []
    all_y_pred = []

    files = [f for f in os.listdir(test_dir) if f.endswith(".json")]
    print(f"Evaluating full system on {len(files)} test samples...")

    for fname in files:
        # ... (rest of the file reading and prediction logic)
        fpath = os.path.join(test_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        # Ground Truth
        raw_labels = data.get('metadata', {}).get('label')
        if not raw_labels or len(raw_labels) != 10:
            continue
            
        y_true = [1 if raw_labels[i] == "False" else 0 for i in range(10)]
        
        # Prediction
        pose_sequence = data.get("pose_sequence")
        # Pre-process pose_sequence as API does
        processed_sequence = []
        for frame in pose_sequence:
            sorted_joints = sorted(frame, key=lambda x: x.get("index", 0))
            frame_data = []
            for j in sorted_joints:
                def safe_float(val, default=0.0):
                    return float(val) if val is not None else default
                
                frame_data.append([
                    safe_float(j.get("x_3d_meters")),
                    safe_float(j.get("y_3d_meters")),
                    safe_float(j.get("z_3d_meters")),
                    safe_float(j.get("visibility"), 1.0)
                ])
            processed_sequence.append(frame_data)
        
        result = engine.predict(processed_sequence)
        
        # The engine returns 'mistakes' (list of names). 
        # We convert it back to a binary vector of length 10.
        y_pred = [0] * 10
        mistakes = result["mistakes"]
        for i, label_name in enumerate(LABELS):
            if label_name in mistakes:
                y_pred[i] = 1
                
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print("\n" + "="*85)
    print(f"{'Label':<15} | {'Type':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 85)

    for i, name in enumerate(LABELS):
        y_t = all_y_true[:, i]
        y_p = all_y_pred[:, i]
        
        l_type = "Rule" if i < 6 else "AI"
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
        
        print(f"{name:<15} | {l_type:<10} | {acc:<10.4f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

    # Overall Subset Accuracy
    subset_acc = np.all(all_y_true == all_y_pred, axis=1).mean()
    print("-" * 85)
    print(f"{'OVERALL FULL-SYSTEM SUBSET ACCURACY:':<65} {subset_acc:.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    evaluate_full_system()
