import numpy as np
from src.config import JOINT_MAP, THRESHOLDS

class VirtualNodeSynthesizer:
    @staticmethod
    def synthesize(pose_seq):
        """
        pose_seq: (frames, 33, features)
        Returns: (frames, 36, features)
        """
        frames, joints, features = pose_seq.shape
        virtual_nodes = np.zeros((frames, 3, features))
        
        # 33: Mid_Hip (23, 24)
        virtual_nodes[:, 0, :] = (pose_seq[:, 23, :] + pose_seq[:, 24, :]) / 2.0
        # 34: Mid_Shoulder (11, 12)
        virtual_nodes[:, 1, :] = (pose_seq[:, 11, :] + pose_seq[:, 12, :]) / 2.0
        # 35: Mid_Ear (7, 8)
        virtual_nodes[:, 2, :] = (pose_seq[:, 7, :] + pose_seq[:, 8, :]) / 2.0
        
        return np.concatenate([pose_seq, virtual_nodes], axis=1)

class SquatStateMachine:
    def __init__(self, fps=30):
        self.fps = fps
        self.reset()

    def reset(self):
        self.phases = {
            "START": [],
            "DESCENT": [],
            "BOTTOM": None, # Index of apex
            "ASCENT": [],
            "FINISH": []
        }

    def analyze(self, mid_hip_y):
        """
        mid_hip_y: (frames,) - vertical Y coordinates
        In MediaPipe, +Y is DOWN.
        """
        self.reset()
        num_frames = len(mid_hip_y)
        if num_frames == 0:
            return self.phases
            
        # Velocity calculation (moving average to reduce noise)
        velocity = np.gradient(mid_hip_y)
        window_size = 5
        velocity_smooth = np.convolve(velocity, np.ones(window_size)/window_size, mode='same')
        
        # 1. Find Bottom: Max Y value
        bottom_idx = np.argmax(mid_hip_y)
        self.phases["BOTTOM"] = int(bottom_idx)
        
        # 2. Detect START -> DESCENT transition
        # Threshold for movement: 0.002 units per frame (roughly 6cm/s at normalized height)
        v_threshold = 0.001 
        start_end = 0
        for i in range(bottom_idx):
            if velocity_smooth[i] > v_threshold:
                start_end = i
                break
        
        # 3. Detect ASCENT -> FINISH transition
        finish_start = num_frames - 1
        for i in range(num_frames - 1, bottom_idx, -1):
            if velocity_smooth[i] < -v_threshold:
                finish_start = i
                break
        
        # Assign phases
        self.phases["START"] = list(range(0, start_end))
        self.phases["DESCENT"] = list(range(start_end, bottom_idx))
        self.phases["ASCENT"] = list(range(bottom_idx + 1, finish_start + 1))
        self.phases["FINISH"] = list(range(finish_start + 1, num_frames))
        
        # Ensure BOTTOM is also categorized if it's a single frame
        # (It's handled separately in InferenceEngine but good for consistency)
        
        return self.phases

class RuleBasedHead:
    def __init__(self):
        self.thresholds = THRESHOLDS

    def get_angle(self, v1, v2):
        """Angle between two vectors in degrees."""
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def evaluate(self, pose_seq, phases):
        """
        pose_seq: (frames, 36, 3)
        phases: dict from SquatStateMachine
        Returns: dict with 'binary', 'values', and 'frame_severity'
        """
        num_frames = pose_seq.shape[0]
        binary = [0] * 6
        values = {}
        frame_severity = np.zeros((num_frames, 6))
        
        # Helper to calculate severity (0 to 1)
        # 0 at threshold, 1 at threshold + 25% (more aggressive transition)
        def calc_severity(val, thresh, scale=0.25):
            if val <= thresh: return 0.0
            return float(min(1.0, (val - thresh) / (thresh * scale + 1e-6)))

        # 1. Head Position
        head_tilts = []
        vertical = np.array([0, 1, 0])
        for f in range(num_frames):
            v_neck = pose_seq[f, 35] - pose_seq[f, 34]
            angle = self.get_angle(v_neck, vertical)
            head_tilts.append(angle)
            frame_severity[f, 0] = calc_severity(angle, self.thresholds["HEAD_TILT"])
        
        avg_head_tilt = np.mean(head_tilts)
        values["Head"] = {"val": float(avg_head_tilt), "threshold": self.thresholds["HEAD_TILT"], "unit": "°"}
        binary[0] = 1 if avg_head_tilt > self.thresholds["HEAD_TILT"] else 0

        # 2. Hip Position
        hip_drops = []
        hip_shifts = []
        active_frames = phases["DESCENT"] + [phases["BOTTOM"]] + phases["ASCENT"]
        for f in range(num_frames):
            if f in active_frames:
                left_hip = pose_seq[f, 23]
                right_hip = pose_seq[f, 24]
                
                drop = abs(left_hip[1] - right_hip[1])
                shift = abs(pose_seq[f, 33, 0] - pose_seq[f, 34, 0])
                
                hip_drops.append(drop)
                hip_shifts.append(shift)
                
                s_drop = calc_severity(drop, self.thresholds["HIP_DROP"])
                s_shift = calc_severity(shift, self.thresholds["HIP_SHIFT"])
                frame_severity[f, 1] = max(s_drop, s_shift)
            
        avg_drop = np.mean(hip_drops) if hip_drops else 0
        avg_shift = np.mean(hip_shifts) if hip_shifts else 0
        is_drop_error = avg_drop > self.thresholds["HIP_DROP"]
        is_shift_error = avg_shift > self.thresholds["HIP_SHIFT"]
        
        if is_drop_error:
            values["Hip"] = {"val": float(avg_drop), "threshold": self.thresholds["HIP_DROP"], "unit": "m"}
        else:
            values["Hip"] = {"val": float(avg_shift), "threshold": self.thresholds["HIP_SHIFT"], "unit": "m"}
        binary[1] = 1 if is_drop_error or is_shift_error else 0

        # 3. Frontal Knee Position
        valgus_scores = []
        for f in range(num_frames):
            if f in active_frames:
                l_valgus = pose_seq[f, 25, 0] - pose_seq[f, 27, 0]
                r_valgus = pose_seq[f, 28, 0] - pose_seq[f, 26, 0]
                v_score = max(l_valgus, r_valgus)
                valgus_scores.append(v_score)
                frame_severity[f, 2] = calc_severity(v_score, self.thresholds["KNEE_VALGUS"])
            
        avg_valgus = np.mean(valgus_scores) if valgus_scores else 0
        values["Frontal Knee"] = {"val": float(avg_valgus), "threshold": self.thresholds["KNEE_VALGUS"], "unit": "m"}
        binary[2] = 1 if avg_valgus > self.thresholds["KNEE_VALGUS"] else 0

        # 4. Tibial Progression Angle
        tibia_angles = []
        descent_frames = phases["DESCENT"] + [phases["BOTTOM"]]
        for f in range(num_frames):
            if f in descent_frames:
                v_tibia = pose_seq[f, 25] - pose_seq[f, 27]
                v_torso = pose_seq[f, 34] - pose_seq[f, 33]
                angle = self.get_angle(v_tibia, v_torso)
                tibia_angles.append(angle)
                frame_severity[f, 3] = calc_severity(angle, self.thresholds["TIBIAL_PARALLEL"])
            
        avg_tibia_angle = np.mean(tibia_angles) if tibia_angles else 0
        values["Tibial Angle"] = {"val": float(avg_tibia_angle), "threshold": self.thresholds["TIBIAL_PARALLEL"], "unit": "°"}
        binary[3] = 1 if avg_tibia_angle > self.thresholds["TIBIAL_PARALLEL"] else 0

        # 5. Foot Position
        setup_frames = pose_seq[phases["START"]]
        ground_y = np.mean(setup_frames[:, [29, 30, 31, 32], 1])
        lifts = []
        for f in range(num_frames):
            lift = np.max(ground_y - pose_seq[f, [29, 30, 31, 32], 1])
            lifts.append(lift)
            frame_severity[f, 4] = calc_severity(lift, self.thresholds["FOOT_LIFT"])
            
        avg_lift = np.mean(lifts)
        values["Foot"] = {"val": float(avg_lift), "threshold": self.thresholds["FOOT_LIFT"], "unit": "m"}
        binary[4] = 1 if avg_lift > self.thresholds["FOOT_LIFT"] else 0

        # 6. Depth
        f_bot = phases["BOTTOM"]
        hip_y = (pose_seq[f_bot, 23, 1] + pose_seq[f_bot, 24, 1]) / 2.0
        knee_y = (pose_seq[f_bot, 25, 1] + pose_seq[f_bot, 26, 1]) / 2.0
        depth_val = knee_y - hip_y
        values["Depth"] = {"val": float(depth_val), "threshold": self.thresholds["DEPTH_OFFSET"], "unit": "m"}
        binary[5] = 1 if depth_val > self.thresholds["DEPTH_OFFSET"] else 0
        
        # Assign depth severity only to frames near the bottom (±5 frames)
        if f_bot is not None:
            depth_sev = calc_severity(depth_val, self.thresholds["DEPTH_OFFSET"])
            for f in range(max(0, f_bot-5), min(num_frames, f_bot+6)):
                frame_severity[f, 5] = depth_sev
            
        raw_metrics = {
            "Head": float(avg_head_tilt),
            "Hip_Drop": float(avg_drop),
            "Hip_Shift": float(avg_shift),
            "Frontal Knee": float(avg_valgus),
            "Tibial Angle": float(avg_tibia_angle),
            "Foot": float(avg_lift),
            "Depth": float(depth_val)
        }
            
        return {"binary": binary, "values": values, "frame_severity": frame_severity, "raw_metrics": raw_metrics}
