export const JOINT_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
  "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
  "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
  "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
  "right_heel", "left_foot_index", "right_foot_index",
  "mid_hip", "mid_shoulder", "mid_ear"
];

export const SKELETON_EDGES = [
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [23, 33], [24, 33], [11, 34], [12, 34], [7, 35], [8, 35], [33, 34], [34, 35]
];

export const LABELS = [
  "Head", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Depth",
  "Thoracic", "Trunk", "Descent", "Ascent"
];

export const LABELS_DATASET = ["Head", "Thoracic", "Trunk", "Hip", "Frontal Knee", "Tibial Angle", "Foot", "Descent", "Depth", "Ascent"];
