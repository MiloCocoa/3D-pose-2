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
        if os.path.exists(report_path):
            report = pd.read_csv(report_path)
            valid_files = set(report[report['status'] == 'PASS']['file'].tolist())
            self.file_list = [f for f in all_files if f in valid_files]
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
                frame_data.append([
                    joint['x_3d_meters'], joint['y_3d_meters'], joint['z_3d_meters'], joint['visibility']
                ])
            pose_seq.append(frame_data)
        
        pose_seq = np.array(pose_seq, dtype=np.float32) # (frames, 33, 4)
        
        # Synthesis
        pose_seq = self.synthesizer.synthesize(pose_seq) # (frames, 36, 4)
        
        # Normalization (Matches Inference)
        # Use height from first 10% of frames as a proxy for SETUP phase
        setup_limit = max(5, int(pose_seq.shape[0] * 0.1))
        setup_heights = np.linalg.norm(pose_seq[:setup_limit, 34, :3] - pose_seq[:setup_limit, 33, :3], axis=1)
        avg_height = np.mean(setup_heights) if len(setup_heights) > 0 else 1.0
        
        for f in range(pose_seq.shape[0]):
            pose_seq[f, :, :3] = (pose_seq[f, :, :3] - pose_seq[f, 33, :3]) / (avg_height + 1e-6)

        # Resample
        pose_seq = self.resample_sequence(pose_seq)
        pose_seq = pose_seq.transpose(1, 0, 2).reshape(36, -1)
        
        # Labels: Only take the 4 postural labels for ST-GCN training
        # Dataset indices for [Thoracic, Trunk, Descent, Ascent] are [1, 2, 7, 9]
        raw_labels = data['metadata']['label'] # List of 10 "True"/"False"
        st_gcn_indices = [1, 2, 7, 9]
        label_vector = [1.0 if raw_labels[i] == "False" else 0.0 for i in st_gcn_indices]
        
        return {
            'pose': torch.tensor(pose_seq, dtype=torch.float32), 
            'label': torch.tensor(label_vector, dtype=torch.float32)
        }
