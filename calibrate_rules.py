import os
import json
import numpy as np
from src.rules.engine import SquatStateMachine, RuleBasedHead, VirtualNodeSynthesizer
from src.data_utils.loader import smooth_sequence
from src.config import DATA_DIR, LABELS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calibrate():
    rule_labels = LABELS[:6] # Head, Hip, Frontal Knee, Tibial Angle, Foot, Depth
    
    y_true_all = {l: [] for l in rule_labels}
    y_pred_all = {l: [] for l in rule_labels}
    
    synthesizer = VirtualNodeSynthesizer()
    state_machine = SquatStateMachine()
    rule_engine = RuleBasedHead()

    folders = [os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "test")]
    total_files = 0
    
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
            
            # Synthesize
            pose_seq = synthesizer.synthesize(pose_seq)
            
            # Outlier Rejection & Smoothing
            pose_seq = smooth_sequence(pose_seq)
            
            # Normalization (matches InferenceEngine)
            setup_limit = max(5, int(pose_seq.shape[0] * 0.1))
            setup_heights = np.linalg.norm(pose_seq[:setup_limit, 34, :3] - pose_seq[:setup_limit, 33, :3], axis=1)
            avg_height = np.mean(setup_heights) if len(setup_heights) > 0 else 1.0
            
            for f in range(pose_seq.shape[0]):
                pose_seq[f, :, :3] = (pose_seq[f, :, :3] - pose_seq[f, 33, :3]) / (avg_height + 1e-6)
                
            # DO NOT resample for rules, evaluate on original frames
            mid_hip_y = pose_seq[:, 33, 1]
            phases = state_machine.analyze(mid_hip_y)
            
            if phases["BOTTOM"] is None:
                continue
                
            result = rule_engine.evaluate(pose_seq[..., :3], phases)
            pred_binary = result["binary"]
            
            for i, label in enumerate(rule_labels):
                # "True" means Pass (0), "False" means Fail (1)
                is_pass = raw_labels[i] == "True"
                y_true = 0 if is_pass else 1
                y_true_all[label].append(y_true)
                y_pred_all[label].append(pred_binary[i])
                
            total_files += 1
                
    print(f"=== Rule-Based Head Calibration Report ===")
    print(f"Evaluated on {total_files} samples.\n")
    
    print(f"{'Label':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    for label in rule_labels:
        y_t = np.array(y_true_all[label])
        y_p = np.array(y_pred_all[label])
        
        if len(y_t) == 0: continue
        
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
        
        print(f"{label:<15} | {acc:<10.4f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        
    print("-" * 65)

if __name__ == "__main__":
    calibrate()