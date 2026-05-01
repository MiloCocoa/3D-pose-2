import os
import json
import numpy as np

data_dir = os.path.join("data", "test-pos-seq-20260311")
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

for fname in files:
    with open(os.path.join(data_dir, fname), 'r') as f:
        data = json.load(f)
    
    pose_seq = []
    for f_idx, frame in enumerate(data['pose_sequence']):
        for j_idx, joint in enumerate(frame):
            for val in [joint['x_3d_meters'], joint['y_3d_meters'], joint['z_3d_meters']]:
                if np.isnan(val) or np.isinf(val):
                    print(f"File {fname}: NaN/Inf found in frame {f_idx}, joint {j_idx}")
                if abs(val) > 100: # Sanity check for meters
                    print(f"File {fname}: Outlier value {val} found in frame {f_idx}, joint {j_idx}")

print("Scan complete.")
