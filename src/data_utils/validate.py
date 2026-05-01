import torch
from src.data_utils.loader import PoseDataset
from src.config import TARGET_FRAMES, NUM_JOINTS, LABELS
import os

def validate_dataset(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist.")
        return False
    
    dataset = PoseDataset(data_dir)
    if len(dataset) == 0:
        print(f"Error: No JSON files found in {data_dir}.")
        return False
    
    print(f"Found {len(dataset)} samples in {data_dir}.")
    
    # Try loading the first sample
    try:
        sample = dataset[0]
        pose = sample['pose']
        label = sample['label']
        
        print(f"Pose shape: {pose.shape} (Expected: ({TARGET_FRAMES}, {NUM_JOINTS}, 3))")
        print(f"Label shape: {label.shape} (Expected: ({len(LABELS)}))")
        
        # Check shapes
        assert pose.shape == (TARGET_FRAMES, NUM_JOINTS, 3)
        assert label.shape == (len(LABELS),)
        
        print("Validation successful!")
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    data_path = os.path.join("data", "test-pos-seq-20260311")
    validate_dataset(data_path)
