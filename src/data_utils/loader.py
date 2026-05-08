import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from src.config import TARGET_FRAMES, NUM_JOINTS, LABELS, LABELS_DATASET, DATA_DIR, VALIDATION_REPORT_PATH, MAX_GAP_SIZE, INPUT_FEATURES
from src.rules.engine import VirtualNodeSynthesizer

class PoseDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, target_frames=TARGET_FRAMES, report_path=VALIDATION_REPORT_PATH):
        self.data_dir = data_dir
        self.target_frames = target_frames
        self.synthesizer = VirtualNodeSynthesizer()
        
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        # If a report path is given and it exists, use it to filter.
        if report_path and os.path.exists(report_path):
            report = pd.read_csv(report_path)
            valid_files = set(report[report['status'] == 'PASS']['file'].tolist())
            self.file_list = [f for f in all_files if f in valid_files]
            if len(self.file_list) == 0:
                print(f"Warning: Validation report filtered out all files in {data_dir}. Falling back to loading all files.")
                self.file_list = all_files
        else:
            self.file_list = all_files

    def __len__(self):
        return len(self.file_list)

    def resample_sequence(self, sequence):
        num_frames = sequence.shape[0]
        num_joints = sequence.shape[1]
        num_features = sequence.shape[2]
        
        if num_frames == self.target_frames:
            return sequence
        
        x = np.arange(num_frames)
        x_new = np.linspace(0, num_frames - 1, self.target_frames)
        
        resampled_sequence = np.zeros((self.target_frames, num_joints, num_features))
        for joint in range(num_joints):
            for dim in range(num_features):
                f = interp1d(x, sequence[:, joint, dim], kind='linear')
                resampled_sequence[:, joint, dim] = f(x_new)
        return resampled_sequence

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        pose_seq = []
        for frame in data['pose_sequence']:
            frame_data = []
            sorted_joints = sorted(frame, key=lambda x: x['index'])
            for joint in sorted_joints:
                def safe_float(val, default=0.0):
                    return float(val) if val is not None else default
                frame_data.append([
                    safe_float(joint.get('x_3d_meters')), 
                    safe_float(joint.get('y_3d_meters')), 
                    safe_float(joint.get('z_3d_meters')), 
                    safe_float(joint.get('visibility'), 1.0)
                ])
            pose_seq.append(frame_data)
        
        pose_seq = np.array(pose_seq, dtype=np.float32) # (frames, 33, 4)
        num_frames = pose_seq.shape[0]
        
        # 1. Synthesis
        pose_seq = self.synthesizer.synthesize(pose_seq) # (frames, 36, 4)
        
        # 2. Outlier Rejection & Smoothing
        pose_seq = smooth_sequence(pose_seq)
        
        # 3. Normalization (Matches Inference)
        setup_limit = max(5, int(num_frames * 0.1))
        setup_heights = np.linalg.norm(pose_seq[:setup_limit, 34, :3] - pose_seq[:setup_limit, 33, :3], axis=1)
        avg_height = np.mean(setup_heights) if len(setup_heights) > 0 else 1.0
        
        for f in range(num_frames):
            pose_seq[f, :, :3] = (pose_seq[f, :, :3] - pose_seq[f, 33, :3]) / (avg_height + 1e-6)

        # 4. Biomechanical Feature Injection (Trunk Angle, Left Knee Angle, Right Knee Angle)
        def compute_angle(v1, v2):
            cos_theta = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
            return np.arccos(np.clip(cos_theta, -1.0, 1.0))

        v_shoulder = pose_seq[:, 34, :3] - pose_seq[:, 33, :3]
        v_vertical = np.zeros_like(v_shoulder)
        v_vertical[:, 1] = -1.0 # MediaPipe Y is down, -1 is up
        trunk_angle = compute_angle(v_shoulder, v_vertical)

        v_lk_hip = pose_seq[:, 23, :3] - pose_seq[:, 25, :3]
        v_lk_ank = pose_seq[:, 27, :3] - pose_seq[:, 25, :3]
        l_knee_angle = compute_angle(v_lk_hip, v_lk_ank)

        v_rk_hip = pose_seq[:, 24, :3] - pose_seq[:, 26, :3]
        v_rk_ank = pose_seq[:, 28, :3] - pose_seq[:, 26, :3]
        r_knee_angle = compute_angle(v_rk_hip, v_rk_ank)

        trunk_angle_b = np.tile(trunk_angle[:, None, None], (1, 36, 1))
        l_knee_angle_b = np.tile(l_knee_angle[:, None, None], (1, 36, 1))
        r_knee_angle_b = np.tile(r_knee_angle[:, None, None], (1, 36, 1))
        
        pose_seq_feat = np.concatenate([pose_seq[:, :, :4], trunk_angle_b, l_knee_angle_b, r_knee_angle_b], axis=2)

        # 5. Phase-Aware Temporal Normalization
        from src.rules.engine import SquatStateMachine
        sm = SquatStateMachine()
        mid_hip_y = pose_seq[:, 33, 1]
        phases = sm.analyze(mid_hip_y)
        bottom_idx = phases.get("BOTTOM")
        
        is_valid_phases = (bottom_idx is not None and 
                          bottom_idx > 5 and 
                          bottom_idx < num_frames - 5 and
                          len(phases.get("DESCENT", [])) > 0)

        if is_valid_phases:
            descent_block = pose_seq_feat[:bottom_idx + 1]
            ascent_block = pose_seq_feat[bottom_idx:]
            
            def local_resample(seq, target):
                if seq.shape[0] < 2: return np.zeros((target, 36, 7))
                x = np.arange(seq.shape[0])
                x_new = np.linspace(0, seq.shape[0] - 1, target)
                res = np.zeros((target, 36, 7))
                for j in range(36):
                    for d in range(7):
                        f_interp = interp1d(x, seq[:, j, d], kind='linear', fill_value="extrapolate")
                        res[:, j, d] = f_interp(x_new)
                return res

            pose_resampled = np.concatenate([
                local_resample(descent_block, 50),
                local_resample(ascent_block, 50)
            ], axis=0)
        else:
            x = np.arange(num_frames)
            x_new = np.linspace(0, num_frames - 1, self.target_frames)
            pose_resampled = np.zeros((self.target_frames, 36, 7))
            for j in range(36):
                for d in range(7):
                    f_interp = interp1d(x, pose_seq_feat[:, j, d], kind='linear', fill_value="extrapolate")
                    pose_resampled[:, j, d] = f_interp(x_new)
        
        pose_resampled = np.nan_to_num(pose_resampled)
        pose_final = pose_resampled.transpose(1, 0, 2).reshape(36, -1)
        
        raw_labels = data['metadata']['label'] 
        st_gcn_indices = [1, 2, 7, 9] 
        label_vector = [1.0 if raw_labels[i] == "False" else 0.0 for i in st_gcn_indices]
        
        return {
            'pose': torch.tensor(pose_final, dtype=torch.float32), 
            'label': torch.tensor(label_vector, dtype=torch.float32)
        }

def smooth_sequence(sequence, max_speed=0.2):
    """
    sequence: (frames, joints, features) in meters
    max_speed: Max allowed movement in meters per frame (~6 m/s at 30fps).
    Detects outliers, marks them as NaN, and interpolates.
    """
    num_frames = sequence.shape[0]
    num_joints = sequence.shape[1]
    
    flat_seq = sequence.reshape(num_frames, -1)
    
    for j in range(num_joints):
        idx_x, idx_y, idx_z = j*4, j*4+1, j*4+2
        for f in range(1, num_frames):
            prev_pt = flat_seq[f-1, [idx_x, idx_y, idx_z]]
            curr_pt = flat_seq[f, [idx_x, idx_y, idx_z]]
            if np.isnan(prev_pt).any():
                continue
            dist = np.linalg.norm(curr_pt - prev_pt)
            if dist > max_speed:
                flat_seq[f, [idx_x, idx_y, idx_z]] = np.nan
    
    df = pd.DataFrame(flat_seq)
    df = df.interpolate(method='linear', limit_direction='both')
    smoothed_seq = df.to_numpy().reshape(num_frames, num_joints, -1).copy()
    
    window = 3
    kernel = np.ones(window) / window
    for j in range(num_joints):
        for dim in range(3): 
            smoothed_seq[:, j, dim] = np.convolve(smoothed_seq[:, j, dim], kernel, mode='same')
            smoothed_seq[0, j, dim] = smoothed_seq[1, j, dim]
            smoothed_seq[-1, j, dim] = smoothed_seq[-2, j, dim]
            
    return smoothed_seq
