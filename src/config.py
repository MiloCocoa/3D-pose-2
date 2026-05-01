# Landmark indices based on pose_map.txt
JOINT_MAP = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
    # Virtual Nodes
    "mid_hip": 33,
    "mid_shoulder": 34,
    "mid_ear": 35
}

# MediaPipe Pose Edges + Virtual Connections
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Virtual Edges
    (23, 33), (24, 33), # Mid-Hip
    (11, 34), (12, 34), # Mid-Shoulder
    (7, 35), (8, 35),   # Mid-Ear
    (33, 34), (34, 35)  # Torso/Neck spine proxy
]

# 6 Rule-Based Labels + 4 ST-GCN Labels
LABELS = [
    "Head", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Depth", # Rules
    "Thoracic", "Trunk", "Descent", "Ascent" # ST-GCN
]

# Mapping labels to joint indices for dynamic highlighting (Heatmap)
# Based on MediaPipe BlazePose indices (0-32) + Virtual (33-35)
LABEL_JOINT_MAP = {
    "Head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 35], # Nose, eyes, ears, mouth, Mid-Ear
    "Hip": [23, 24, 33], # Left/Right Hip, Mid-Hip
    "Frontal Knee": [25, 26], # Left/Right Knee
    "Tibial Angle": [25, 26, 27, 28], # Knees and Ankles
    "Foot": [27, 28, 29, 30, 31, 32], # Ankles, Heels, Toes
    "Depth": [23, 24, 25, 26, 33], # Hips and Knees
    "Thoracic": [11, 12, 34], # Shoulders, Mid-Shoulder
    "Trunk": [11, 12, 23, 24, 33, 34], # Shoulders and Hips
    "Descent": [23, 24, 25, 26, 27, 28], # Hips, Knees, Ankles
    "Ascent": [23, 24, 25, 26, 27, 28] # Hips, Knees, Ankles
}

# Labels in the order they appear in the dataset
LABELS_DATASET = ["Head", "Thoracic", "Trunk", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Descent", "Depth", "Ascent"]

# Rule Thresholds (Sane Defaults)
THRESHOLDS = {
    "HEAD_TILT": 15.0,      # Degrees
    "HIP_TILT": 5.0,        # Degrees
    "HIP_SHIFT": 0.05,      # Normalized units (relative to shoulder width)
    "KNEE_VALGUS": 0.02,    # Normalized units
    "TIBIAL_PARALLEL": 10.0, # Degrees
    "FOOT_LIFT": 0.02,      # Normalized units
    "DEPTH_OFFSET": 0.05    # Normalized units (Hip Y relative to Knee Y)
}

THRESHOLD = 0.5
TARGET_FRAMES = 100
# Path Configurations
DATA_DIR = "data/test-pos-seq-20260311"
VALIDATION_REPORT_PATH = "data_validation_report.csv"
MODEL_SAVE_DIR = "models"
MODEL_NAME = "multi_label_gcn_v2.pth"

# Model Hyperparameters
NUM_JOINTS = 36 # 33 + 3 Virtual
INPUT_FEATURES = 4 # X, Y, Z, Visibility
HIDDEN_CHANNELS = 128
NUM_GCN_BLOCKS = 3
DROPOUT = 0.0 
LEARNING_RATE = 1e-4
BATCH_SIZE = 10 
EPOCHS = 500
MAX_GAP_SIZE = 10 

# Joint Symmetry for Horizontal Flip Augmentation
SYMMETRY_MAP = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

