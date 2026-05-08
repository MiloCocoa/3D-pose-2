import os
import json
import shutil
import random
from collections import defaultdict

from src.config import DATA_DIR

def main():
    base_dir = DATA_DIR
    all_data_dir = os.path.join(base_dir, "all-data")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Clean existing train/test to avoid mixing
    for folder in [train_dir, test_dir]:
        for f in os.listdir(folder):
            if f.endswith(".json"):
                os.remove(os.path.join(folder, f))

    files = [f for f in os.listdir(all_data_dir) if f.endswith(".json")]
    
    # We care about all 10 labels for stratification
    # LABELS_DATASET = ["Head", "Thoracic", "Trunk", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Descent", "Depth", "Ascent"]
    target_indices = {
        'Head': 0, 'Thoracic': 1, 'Trunk': 2, 'Hip': 3, 
        'Frontal Knee': 4, 'Tibial Angle': 5, 'Foot': 6, 
        'Descent': 7, 'Depth': 8, 'Ascent': 9
    }
    
    file_labels = {}
    label_counts = defaultdict(int)
    
    for f in files:
        path = os.path.join(all_data_dir, f)
        with open(path, 'r') as file:
            data = json.load(file)
            raw_labels = data['metadata']['label']
            
            # Extract 1 if Fail (False), 0 if Pass (True)
            fails = {name: (1 if raw_labels[idx] == "False" else 0) for name, idx in target_indices.items()}
            file_labels[f] = fails
            
            for name, is_fail in fails.items():
                if is_fail:
                    label_counts[name] += 1

    print("Total Fails across all data:")
    for name, count in label_counts.items():
        print(f"  {name}: {count}")

    # Greedy Multi-label Stratification
    # Target split ratio: 80% Train, 20% Test
    test_ratio = 0.2
    
    train_files = []
    test_files = []
    
    train_fail_counts = defaultdict(int)
    test_fail_counts = defaultdict(int)

    # Sort files by total number of fails (descending)
    random.seed(42)
    random.shuffle(files)
    files.sort(key=lambda x: sum(file_labels[x].values()), reverse=True)

    for f in files:
        fails = file_labels[f]
        
        # If this file has no fails, allocate based on pure size ratio
        if sum(fails.values()) == 0:
            if len(test_files) < (len(train_files) + len(test_files)) * test_ratio:
                test_files.append(f)
            else:
                train_files.append(f)
            continue

        # For files with fails, figure out which split needs these fails more
        test_preference_score = 0
        for name, is_fail in fails.items():
            if is_fail:
                current_total = train_fail_counts[name] + test_fail_counts[name]
                if current_total == 0:
                    test_preference_score += test_ratio
                else:
                    current_ratio = test_fail_counts[name] / current_total
                    test_preference_score += (test_ratio - current_ratio)
        
        if test_preference_score > 0:
            test_files.append(f)
            for name, is_fail in fails.items():
                test_fail_counts[name] += is_fail
        else:
            train_files.append(f)
            for name, is_fail in fails.items():
                train_fail_counts[name] += is_fail

    # Move files
    for f in train_files:
        shutil.copy(os.path.join(all_data_dir, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(all_data_dir, f), os.path.join(test_dir, f))

    print(f"\nSplit complete!")
    print(f"Train: {len(train_files)} files")
    for name in target_indices.keys():
        print(f"  {name} Fails: {train_fail_counts[name]}")
        
    print(f"\nTest: {len(test_files)} files")
    for name in target_indices.keys():
        print(f"  {name} Fails: {test_fail_counts[name]}")

if __name__ == "__main__":
    main()
