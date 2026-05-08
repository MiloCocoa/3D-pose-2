import os
import json
import shutil
import random
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

# Constants from project
SYMMETRY_MAP = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

def mirror_pose(pose_sequence):
    mirrored = []
    for frame in pose_sequence:
        new_frame = []
        for joint in frame:
            j = joint.copy()
            x = j.get('x_3d_meters')
            j['x_3d_meters'] = (float(x) * -1) if x is not None else 0.0
            new_frame.append(j)
        
        # Swap L/R based on symmetry map
        final_frame = sorted(new_frame, key=lambda x: x['index'])
        for left, right in SYMMETRY_MAP:
            l_val = final_frame[left].copy()
            r_val = final_frame[right].copy()
            # Swap everything except index, handling Nones
            for key in ['x_3d_meters', 'y_3d_meters', 'z_3d_meters', 'visibility']:
                lv = l_val.get(key)
                rv = r_val.get(key)
                final_frame[left][key] = float(rv) if rv is not None else 0.0
                final_frame[right][key] = float(lv) if lv is not None else 0.0
        mirrored.append(final_frame)
    return mirrored

def add_jitter(pose_sequence, std=0.005):
    jittered = []
    for frame in pose_sequence:
        new_frame = []
        for joint in frame:
            j = joint.copy()
            for key in ['x_3d_meters', 'y_3d_meters', 'z_3d_meters']:
                val = j.get(key)
                j[key] = (float(val) + np.random.normal(0, std)) if val is not None else 0.0
            new_frame.append(j)
        jittered.append(new_frame)
    return jittered

def scale_time(pose_sequence, factor=1.1):
    num_frames = len(pose_sequence)
    new_num_frames = int(num_frames * factor)
    
    # Flatten joints for interpolation
    flat_data = []
    for frame in pose_sequence:
        sorted_joints = sorted(frame, key=lambda x: x['index'])
        frame_vec = []
        for j in sorted_joints:
            for key in ['x_3d_meters', 'y_3d_meters', 'z_3d_meters', 'visibility']:
                val = j.get(key)
                frame_vec.append(float(val) if val is not None else 0.0)
        flat_data.append(frame_vec)
    
    flat_data = np.array(flat_data)
    x = np.arange(num_frames)
    x_new = np.linspace(0, num_frames - 1, new_num_frames)
    
    new_flat_data = np.zeros((new_num_frames, flat_data.shape[1]))
    for i in range(flat_data.shape[1]):
        f = interp1d(x, flat_data[:, i], kind='linear', fill_value="extrapolate")
        new_flat_data[:, i] = f(x_new)
        
    # Reconstruct
    new_sequence = []
    for f in range(new_num_frames):
        new_frame = []
        for i in range(33):
            new_frame.append({
                "index": i,
                "x_3d_meters": float(new_flat_data[f, i*4]),
                "y_3d_meters": float(new_flat_data[f, i*4 + 1]),
                "z_3d_meters": float(new_flat_data[f, i*4 + 2]),
                "visibility": float(new_flat_data[f, i*4 + 3])
            })
        new_sequence.append(new_frame)
    return new_sequence

def main():
    source_dir = "data/test-pos-seq-v4-balanced/all-data"
    output_dir = "data/test-pos-seq-v4-balanced"
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.endswith(".json")]
    
    # Multi-label Stratification (80/20)
    target_indices = {
        'Head': 0, 'Thoracic': 1, 'Trunk': 2, 'Hip': 3, 
        'Frontal Knee': 4, 'Tibial Angle': 5, 'Foot': 6, 
        'Descent': 7, 'Depth': 8, 'Ascent': 9
    }
    
    file_labels = {}
    for f in files:
        with open(os.path.join(source_dir, f), 'r') as file:
            data = json.load(file)
            raw_labels = data['metadata']['label']
            file_labels[f] = [1 if raw_labels[idx] == "False" else 0 for idx in range(10)]

    random.seed(42)
    random.shuffle(files)
    files.sort(key=lambda x: sum(file_labels[x]), reverse=True)

    train_files, test_files = [], []
    train_counts = np.zeros(10)
    test_counts = np.zeros(10)
    test_ratio = 0.2

    for f in files:
        labels = np.array(file_labels[f])
        if sum(labels) == 0:
            if len(test_files) < (len(train_files) + len(test_files)) * test_ratio:
                test_files.append(f)
            else:
                train_files.append(f)
        else:
            test_pref = 0
            for i in range(10):
                if labels[i] == 1:
                    total = train_counts[i] + test_counts[i]
                    test_pref += (test_ratio - (test_counts[i] / total if total > 0 else test_ratio))
            
            if test_pref > 0:
                test_files.append(f)
                test_counts += labels
            else:
                train_files.append(f)
                train_counts += labels

    print(f"Original Train Fails: {train_counts.astype(int)}")
    
    # --- AUGMENTATION PHASE (Train Only) ---
    print("\nStarting Augmentation for balancing...")
    
    for f in train_files:
        fpath = os.path.join(source_dir, f)
        with open(fpath, 'r') as file:
            data = json.load(file)
        
        # Save original
        with open(os.path.join(train_dir, f), 'w') as out:
            json.dump(data, out)
            
        labels = np.array(file_labels[f])
        
        # Determine if this is a "Rare" class sequence (fails in minority labels)
        # Minorities: Head(0), Thoracic(1), Trunk(2), Tibial(5), Depth(8), Ascent(9)
        is_rare = any(labels[[0, 1, 2, 5, 8, 9]] == 1)
        
        if is_rare:
            # 1. Mirroring
            mirrored_data = data.copy()
            mirrored_data['pose_sequence'] = mirror_pose(data['pose_sequence'])
            # Note: We assume labels stay the same for mirroring (postural faults are symmetrical)
            with open(os.path.join(train_dir, f"mirrored_{f}"), 'w') as out:
                json.dump(mirrored_data, out)
                
            # 2. Temporal Stretching (Slow)
            slow_data = data.copy()
            slow_data['pose_sequence'] = scale_time(data['pose_sequence'], 1.2)
            with open(os.path.join(train_dir, f"slow_{f}"), 'w') as out:
                json.dump(slow_data, out)
                
            # 3. Temporal Compression (Fast)
            fast_data = data.copy()
            fast_data['pose_sequence'] = scale_time(data['pose_sequence'], 0.8)
            with open(os.path.join(train_dir, f"fast_{f}"), 'w') as out:
                json.dump(fast_data, out)

        # 4. Add Robustness (Jitter) to 30% of ALL training files
        if random.random() < 0.3:
            jitter_data = data.copy()
            jitter_data['pose_sequence'] = add_jitter(data['pose_sequence'])
            with open(os.path.join(train_dir, f"jitter_{f}"), 'w') as out:
                json.dump(jitter_data, out)

    # Copy Test (Untouched)
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

    # Recalculate Final Training Stats
    final_train_counts = np.zeros(10)
    final_files = [f for f in os.listdir(train_dir) if f.endswith(".json")]
    for f in final_files:
        with open(os.path.join(train_dir, f), 'r') as file:
            d = json.load(file)
            l = d['metadata']['label']
            for i in range(10):
                if l[i] == "False": final_train_counts[i] += 1
                
    print(f"\nAugmentation Complete!")
    print(f"Final Train: {len(final_files)} files")
    print(f"Final Train Fails: {final_train_counts.astype(int)}")
    print(f"Test: {len(test_files)} files (Untouched)")

if __name__ == "__main__":
    main()
