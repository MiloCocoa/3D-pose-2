# Dataset and Skeleton Description

This document provides a detailed overview of the 3D pose dataset and skeleton structure used in the 3D Barbell Squat Form Correction project.

## 1. Skeleton Structure (19 Joints)
The project uses a 19-joint skeleton extracted from a raw 25-joint Kinect-style format. The coordinate system is **Y-up** (vertical).

### Joint Mapping (0-indexed)
Based on the implementation in `model.py` and `metrics.py`, the joints are mapped as follows:

| Index | Body Part | Description |
| :--- | :--- | :--- |
| **0** | **Base Spine** | Lower back/Sacrum area |
| **1** | **Mid Spine** | Chest/Thoracic area |
| **2** | **Right Shoulder** | |
| **3** | **Right Elbow** | |
| **4** | **Right Wrist** | |
| **5** | **Left Shoulder** | |
| **6** | **Left Elbow** | |
| **7** | **Left Wrist** | |
| **8** | **Hip Center** | Root/Midpoint between hips |
| **9** | **Left Hip** | |
| **10** | **Left Knee** | |
| **11** | **Left Ankle** | |
| **12** | **Right Hip** | |
| **13** | **Right Knee** | |
| **14** | **Right Ankle** | |
| **15** | **Right Toe** | |
| **17** | **Left Heel** | |
| **18** | **Left Toe** | |
| **19** | **Head/Neck** | Top of the skeleton |

## 2. Bone Connections (Graph Edges)
The Graph Convolutional Network (GCN) uses the following connections to define the skeleton topology:
- **Torso:** [0, 1], [1, 8]
- **Shoulders/Arms:** [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7]
- **Hips/Legs:** [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]
- **Feet:** [11, 17], [11, 18], [14, 15]

## 3. Data Format & Preprocessing
- **Raw Data:** Stored in `data_3D.pickle`.
- **Input Shape:** `(Batch, 57, 100)`.
  - **57 Features:** 19 joints × 3 coordinates (X, Y, Z).
  - **100 Frames:** Every rep is temporally resampled to a fixed length of 100 frames using linear interpolation.
- **Coordinates:** Raw relative 3D coordinates. In the UI, these are converted to real-world units (e.g., **cm**) using the vertical distance between the midpoint of the ankles and the midpoint of the shoulders as a reference.
- **Normalization:** The `normalize_pose_sequence` function (in visualization scripts) anchors the skeleton by centering the midpoint of the ankles at `(0, 0, 0)` in the first frame.

## 4. Dataset Labeling
- **Mistake Classification:** Labels 2–12 represent specific form mistakes (e.g., "Butt Wink," "Bar Tilt").
- **Correct Pose:** Label 1 represents a "perfect" repetition.
- **Pairing:** The system pairs each incorrect repetition with a corresponding correct repetition from the same subject to train the pose correction (regression) branch.
