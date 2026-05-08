import os
import json
import numpy as np
from src.rules.engine import SquatStateMachine, RuleBasedHead, VirtualNodeSynthesizer
from src.data_utils.loader import smooth_sequence
from src.config import DATA_DIR, LABELS
from sklearn.metrics import precision_recall_fscore_support

def find_thresholds():
    rule_labels = LABELS[:6]
    
    # Store ground truth and raw calculated metrics
    data_records = []
    
    synthesizer = VirtualNodeSynthesizer()
    state_machine = SquatStateMachine()
    rule_engine = RuleBasedHead()

    folders = [os.path.join(DATA_DIR, "train")]
    
    for folder in folders:
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if not fname.endswith(".json"): continue
            
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
                
            raw_labels = data.get('metadata', {}).get('label')
            if not raw_labels or len(raw_labels) < 6:
                continue
            
            pose_seq = []
            for frame in data['pose_sequence']:
                frame_data = []
                for joint in sorted(frame, key=lambda x: x['index']):
                    frame_data.append([joint['x_3d_meters'], joint['y_3d_meters'], joint['z_3d_meters'], joint['visibility']])
                pose_seq.append(frame_data)
            
            if not pose_seq:
                continue
                
            pose_seq = np.array(pose_seq, dtype=np.float32)
            pose_seq = synthesizer.synthesize(pose_seq)
            
            # Outlier Rejection & Smoothing
            pose_seq = smooth_sequence(pose_seq)
            
            setup_limit = max(5, int(pose_seq.shape[0] * 0.1))
            setup_heights = np.linalg.norm(pose_seq[:setup_limit, 34, :3] - pose_seq[:setup_limit, 33, :3], axis=1)
            avg_height = np.mean(setup_heights) if len(setup_heights) > 0 else 1.0
            
            for f in range(pose_seq.shape[0]):
                pose_seq[f, :, :3] = (pose_seq[f, :, :3] - pose_seq[f, 33, :3]) / (avg_height + 1e-6)
                
            mid_hip_y = pose_seq[:, 33, 1]
            phases = state_machine.analyze(mid_hip_y)
            
            if phases["BOTTOM"] is None:
                continue
                
            result = rule_engine.evaluate(pose_seq[..., :3], phases)
            raw_metrics = result.get("raw_metrics", {})
            
            # Ground truth: 1 for Fail, 0 for Pass
            y_true = [1 if raw_labels[i] == "False" else 0 for i in range(6)]
            
            data_records.append({
                "y_true": y_true,
                "raw_metrics": raw_metrics
            })
            
    print(f"Loaded {len(data_records)} valid sequences for threshold tuning.\n")
    
    # Define sweep ranges for each metric
    # tuple: (metric_key, min_val, max_val, steps)
    sweeps = {
        "Head": ("Head", 0.0, 45.0, 450),
        "Hip": (["Hip_Drop", "Hip_Shift"], 0.0, 0.3, 300), 
        "Frontal Knee": ("Frontal Knee", 0.0, 0.2, 200),
        "Tibial Angle": ("Tibial Angle", 0.0, 30.0, 300),
        "Foot": ("Foot", 0.0, 0.15, 150),
        "Depth": ("Depth", -0.2, 0.2, 400)
    }

    print(f"{'Label':<15} | {'Best Thresh':<15} | {'Best F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 70)

    for i, label in enumerate(rule_labels):
        metric_info = sweeps[label]
        metric_keys = metric_info[0] if isinstance(metric_info[0], list) else [metric_info[0]]
        min_v, max_v, steps = metric_info[1], metric_info[2], metric_info[3]
        
        y_true_label = np.array([rec["y_true"][i] for rec in data_records])
        
        if np.sum(y_true_label) == 0:
            print(f"{label:<15} | {'No Fails':<15} | {'-':<10} | {'-':<10} | {'-':<10}")
            continue

        # Vectorize metrics extraction
        vals = np.array([[rec["raw_metrics"].get(k, 0) for k in metric_keys] for rec in data_records])
        max_vals = np.max(vals, axis=1)

        best_f1 = -1
        best_thresh = None
        best_p = 0
        best_r = 0
        
        # Fast numpy calculation to replace sklearn overhead
        for thresh in np.linspace(min_v, max_v, steps):
            y_pred = (max_vals > thresh).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true_label == 1))
            fp = np.sum((y_pred == 1) & (y_true_label == 0))
            fn = np.sum((y_pred == 0) & (y_true_label == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_p = p
                best_r = r
                
        # Format threshold string
        t_str = f"{best_thresh:.3f}" if best_thresh is not None else "N/A"
        print(f"{label:<15} | {t_str:<15} | {best_f1:<10.4f} | {best_p:<10.4f} | {best_r:<10.4f}")

if __name__ == "__main__":
    find_thresholds()