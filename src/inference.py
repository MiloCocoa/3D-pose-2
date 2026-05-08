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
        self.input_features = 4 # Default to legacy
        
        # Load GCN (only 4 labels now)
        try:
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                
                if 'spatial_backbone.0.gcn.lin.weight' in state_dict:
                    if 'descent_backbone.0.gcn.lin.weight' in state_dict:
                        self.input_features = 7 # V5 Segment-Aware (4 + 3)
                    else:
                        self.input_features = 10 # V4-1 Multi-Stream (4 + 6)
                    print(f"Detected Multi-Stream model input features: {self.input_features}")
                else:
                    first_layer_weight = state_dict.get('backbone.0.gcn.lin.weight') 
                    if first_layer_weight is not None:
                        in_channels = first_layer_weight.shape[1]
                        self.input_features = in_channels // TARGET_FRAMES
                        print(f"Detected Single-Stream model input features: {self.input_features}")
                
                self.model = MultiLabelGCN(in_channels=TARGET_FRAMES * self.input_features, num_labels=4).to(self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded 4-label model from {model_path}")
            else:
                from src.config import INPUT_FEATURES
                self.input_features = INPUT_FEATURES
                self.model = MultiLabelGCN(in_channels=TARGET_FRAMES * self.input_features, num_labels=4).to(self.device)
                print(f"Warning: No model path found. Initializing new model with {self.input_features} features.")
        except Exception as e:
            print(f"Error loading model: {e}")
            from src.config import INPUT_FEATURES
            self.input_features = INPUT_FEATURES
            self.model = MultiLabelGCN(in_channels=TARGET_FRAMES * self.input_features, num_labels=4).to(self.device)
        
        self.model.eval()

    def resample_sequence(self, sequence, target_frames=TARGET_FRAMES):
        num_frames = sequence.shape[0]
        num_joints = sequence.shape[1]
        num_features = sequence.shape[2]
        if num_frames == target_frames:
            return sequence
        
        x = np.arange(num_frames)
        x_new = np.linspace(0, num_frames - 1, target_frames)
        
        resampled_sequence = np.zeros((target_frames, num_joints, num_features))
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

        # Feature Injection if needed
        if self.input_features == 7:
            def compute_angle(v1, v2):
                cos_theta = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
                return np.arccos(np.clip(cos_theta, -1.0, 1.0))

            v_shoulder = norm_pose[:, 34, :3] - norm_pose[:, 33, :3]
            v_vertical = np.zeros_like(v_shoulder)
            v_vertical[:, 1] = -1.0 # MediaPipe Y is down, -1 is up
            trunk_angle = compute_angle(v_shoulder, v_vertical)

            v_lk_hip = norm_pose[:, 23, :3] - norm_pose[:, 25, :3]
            v_lk_ank = norm_pose[:, 27, :3] - norm_pose[:, 25, :3]
            l_knee_angle = compute_angle(v_lk_hip, v_lk_ank)

            v_rk_hip = norm_pose[:, 24, :3] - norm_pose[:, 26, :3]
            v_rk_ank = norm_pose[:, 28, :3] - norm_pose[:, 26, :3]
            r_knee_angle = compute_angle(v_rk_hip, v_rk_ank)

            trunk_angle_b = np.tile(trunk_angle[:, None, None], (1, 36, 1))
            l_knee_angle_b = np.tile(l_knee_angle[:, None, None], (1, 36, 1))
            r_knee_angle_b = np.tile(r_knee_angle[:, None, None], (1, 36, 1))
            
            ai_input_pose = np.concatenate([norm_pose[:, :, :4], trunk_angle_b, l_knee_angle_b, r_knee_angle_b], axis=2)
        elif self.input_features == 10:
            velocities = np.zeros((num_frames, 36, 3), dtype=np.float32)
            velocities[1:] = norm_pose[1:, :, :3] - norm_pose[:-1, :, :3]
            accelerations = np.zeros((num_frames, 36, 3), dtype=np.float32)
            accelerations[1:] = velocities[1:] - velocities[:-1]
            ai_input_pose = np.concatenate([norm_pose, velocities, accelerations], axis=2)
        else:
            ai_input_pose = norm_pose

        # Phase-Aware Resampling (if valid)
        bottom_idx = phases.get("BOTTOM")
        is_valid_phases = (bottom_idx is not None and 
                          bottom_idx > 5 and 
                          bottom_idx < num_frames - 5 and
                          len(phases.get("DESCENT", [])) > 0)

        if is_valid_phases and self.input_features >= 7: # Use phase-aware for v4/v5
            descent_block = ai_input_pose[:bottom_idx + 1]
            ascent_block = ai_input_pose[bottom_idx:]
            
            resampled_pose = np.concatenate([
                self.resample_sequence(descent_block, 50),
                self.resample_sequence(ascent_block, 50)
            ], axis=0)
        else:
            resampled_pose = self.resample_sequence(ai_input_pose, TARGET_FRAMES) # (100, 36, input_features)

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

        # Sanitization: Ensure all floats are JSON compliant (No NaN or Inf)
        def safe_float(v):
            if isinstance(v, (float, np.float32, np.float64)):
                return float(v) if np.isfinite(v) else 0.0
            return v

        sanitized_rule_values = {}
        for k, v in rule_values.items():
            sanitized_rule_values[k] = {
                "val": safe_float(v["val"]),
                "threshold": safe_float(v["threshold"]),
                "unit": v["unit"]
            }

        sanitized_confidences = {k: safe_float(v) for k, v in confidences.items()}
        sanitized_heatmap = np.nan_to_num(joint_heatmap, nan=0.0, posinf=1.0, neginf=0.0).tolist()
        
        return {
            "mistakes": mistakes,
            "confidences": sanitized_confidences,
            "rule_values": sanitized_rule_values,
            "phase_per_frame": phase_per_frame,
            "joint_heatmap": sanitized_heatmap,
            "phases": {k: (v if isinstance(v, int) else len(v)) for k, v in phases.items()}
        }
