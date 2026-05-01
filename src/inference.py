import torch
import numpy as np
from src.model.gcn import MultiLabelGCN
from src.config import LABELS, THRESHOLD, TARGET_FRAMES, NUM_JOINTS, JOINT_MAP, LABEL_JOINT_MAP
from src.rules.engine import VirtualNodeSynthesizer, SquatStateMachine, RuleBasedHead
from scipy.interpolate import interp1d
import os

class InferenceEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.synthesizer = VirtualNodeSynthesizer()
        self.state_machine = SquatStateMachine()
        self.rule_head = RuleBasedHead()
        
        # Load GCN (only 4 labels now)
        try:
            self.model = MultiLabelGCN(num_labels=4).to(self.device)
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                if state_dict['classifier.weight'].shape[0] == 4:
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded 4-label model from {model_path}")
                else:
                    print(f"Warning: Checkpoint at {model_path} has {state_dict['classifier.weight'].shape[0]} labels, but 4 expected.")
            else:
                print("Warning: No model path provided or model not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = MultiLabelGCN(num_labels=4).to(self.device)
        
        self.model.eval()

    def resample_sequence(self, sequence):
        num_frames = sequence.shape[0]
        num_joints = sequence.shape[1]
        num_features = sequence.shape[2]
        if num_frames == TARGET_FRAMES:
            return sequence
        
        x = np.arange(num_frames)
        x_new = np.linspace(0, num_frames - 1, TARGET_FRAMES)
        
        resampled_sequence = np.zeros((TARGET_FRAMES, num_joints, num_features))
        for joint in range(num_joints):
            for dim in range(num_features):
                f = interp1d(x, sequence[:, joint, dim], kind='linear', fill_value="extrapolate")
                resampled_sequence[:, joint, dim] = f(x_new)
        
        resampled_sequence = np.nan_to_num(resampled_sequence)
        return resampled_sequence

    def predict(self, pose_seq):
        """
        pose_seq: list of frames, each frame is list of joints with [x, y, z]
        """
        pose_seq = np.array(pose_seq, dtype=np.float32) # (original_frames, 33, 4)
        num_frames = pose_seq.shape[0]
        
        # 1. Virtual Node Synthesis
        if pose_seq.shape[2] == 3:
            visibility = np.ones((num_frames, pose_seq.shape[1], 1), dtype=np.float32)
            pose_seq_4d = np.concatenate([pose_seq, visibility], axis=2)
        else:
            pose_seq_4d = pose_seq
            
        pose_seq_36 = self.synthesizer.synthesize(pose_seq_4d) # (original_frames, 36, 4)
        
        # 2. State Machine (Phase Detection)
        mid_hip_y = pose_seq_36[:, 33, 1]
        phases = self.state_machine.analyze(mid_hip_y)
        
        # 3. Rule-Based Head (6 Labels)
        rule_output = self.rule_head.evaluate(pose_seq_36[:, :, :3], phases)
        rule_results = rule_output["binary"]
        rule_values = rule_output["values"]
        rule_frame_severity = rule_output["frame_severity"] # (frames, 6)
        
        # 4. ST-GCN Head (4 Labels)
        setup_frames = pose_seq_36[phases["START"]]
        heights = np.linalg.norm(setup_frames[:, 34, :3] - setup_frames[:, 33, :3], axis=1)
        avg_height = np.mean(heights) if len(heights) > 0 else 1.0
        
        norm_pose = pose_seq_36.copy()
        for f in range(norm_pose.shape[0]):
            norm_pose[f, :, :3] = (norm_pose[f, :, :3] - norm_pose[f, 33, :3]) / (avg_height + 1e-6)

        resampled_pose = self.resample_sequence(norm_pose) # (100, 36, 4)
        gcn_input = resampled_pose.transpose(1, 0, 2).reshape(36, -1)
        input_tensor = torch.tensor(gcn_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        gcn_results = [1 if p > THRESHOLD else 0 for p in probs]
        
        # 5. Dynamic Highlighting (Heatmap)
        # Combine rule_frame_severity and GCN probs
        # We need a per-frame severity for AI labels too
        ai_frame_severity = np.zeros((num_frames, 4))
        for i, p in enumerate(probs):
            if p > THRESHOLD:
                label_name = LABELS[6+i]
                # Apply based on phase
                active_indices = []
                if label_name == "Descent":
                    active_indices = phases["DESCENT"]
                elif label_name == "Ascent":
                    active_indices = phases["ASCENT"]
                else: # Thoracic, Trunk
                    active_indices = phases["DESCENT"] + [phases["BOTTOM"]] + phases["ASCENT"]
                
                for idx in active_indices:
                    ai_frame_severity[idx, i] = float(p)

        # Merge all severities (6 rules + 4 AI)
        all_label_severities = np.concatenate([rule_frame_severity, ai_frame_severity], axis=1) # (frames, 10)
        
        # Map labels to joints
        # joint_heatmap: (frames, 36)
        joint_heatmap = np.zeros((num_frames, 36))
        for i, label_name in enumerate(LABELS):
            affected_joints = LABEL_JOINT_MAP.get(label_name, [])
            for f in range(num_frames):
                sev = all_label_severities[f, i]
                if sev > 0:
                    for j_idx in affected_joints:
                        # Accumulate using min(1.0, sum)
                        joint_heatmap[f, j_idx] = min(1.0, joint_heatmap[f, j_idx] + sev)

        # 6. Final Aggregation
        final_binary = rule_results + gcn_results
        
        phase_per_frame = ["NONE"] * num_frames
        for phase_name, indices in phases.items():
            if phase_name == "BOTTOM":
                if indices is not None:
                    phase_per_frame[indices] = "BOTTOM"
            else:
                for idx in indices:
                    phase_per_frame[idx] = phase_name

        mistakes = []
        confidences = {}
        for i, val in enumerate(final_binary):
            label_name = LABELS[i]
            if val == 1:
                mistakes.append(label_name)
            
            if i < 6:
                confidences[label_name] = float(val)
            else:
                confidences[label_name] = float(probs[i-6])
        
        return {
            "mistakes": mistakes,
            "confidences": confidences,
            "rule_values": rule_values,
            "phase_per_frame": phase_per_frame,
            "joint_heatmap": joint_heatmap.tolist(), # Send to frontend
            "phases": {k: (v if isinstance(v, int) else len(v)) for k, v in phases.items()}
        }
