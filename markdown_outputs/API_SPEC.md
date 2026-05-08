# API Specification: 3D Pose Analysis

This document defines the JSON communication format between the Frontend (React) and the Backend (FastAPI) for the 3D Squat Form Correction system.

## Endpoint: `POST /analyze`

Analyzes a 3D pose sequence to detect squat phases, geometric rule violations, and postural mistakes.

### 1. Request Format (Input)

The request body should be a JSON object containing a `pose_sequence` array.

```json
{
  "pose_sequence": [
    [ 
      {
        "index": 0,
        "x_3d_meters": 0.0449,
        "y_3d_meters": -0.6687,
        "z_3d_meters": 2.2730,
        "visibility": 0.9999
      },
      "..." 
    ],
    "..."
  ]
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `pose_sequence` | `Array<Array<Object>>` | An array of frames, where each frame contains 33 joint objects. |
| `index` | `Integer` | MediaPipe joint index (0-32). |
| `x_3d_meters` | `Float` | X coordinate in meters (3D space). |
| `y_3d_meters` | `Float` | Y coordinate in meters (+Y is DOWN). |
| `z_3d_meters` | `Float` | Z coordinate in meters. |
| `visibility` | `Float` | Confidence score (0.0 to 1.0). Defaults to 1.0 if omitted. |

---

### 2. Response Format (Output)

Returns a comprehensive analysis of the motion, including frame-by-frame severity heatmaps.

#### Example Response
```json
{
  "mistakes": ["Depth", "Trunk"],
  "confidences": {
    "Head": 0.0,
    "Hip": 0.0,
    "Frontal Knee": 0.0,
    "Tibial Angle": 0.0,
    "Foot": 0.0,
    "Depth": 1.0,
    "Thoracic": 0.12,
    "Trunk": 0.88,
    "Descent": 0.05,
    "Ascent": 0.02
  },
  "rule_values": {
    "Head": { "val": 4.2, "threshold": 15.0, "unit": "°" },
    "Hip": { "val": 0.012, "threshold": 0.05, "unit": "m" },
    "Frontal Knee": { "val": 0.008, "threshold": 0.02, "unit": "m" },
    "Tibial Angle": { "val": 3.5, "threshold": 10.0, "unit": "°" },
    "Foot": { "val": 0.002, "threshold": 0.02, "unit": "m" },
    "Depth": { "val": 0.115, "threshold": 0.05, "unit": "m" }
  },
  "phase_per_frame": ["DESCENT", "BOTTOM", "ASCENT"],
  "joint_heatmap": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.88, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.0]
  ],
  "phases": {
    "START": 10,
    "DESCENT": 1,
    "BOTTOM": 1,
    "ASCENT": 1,
    "FINISH": 5
  }
}
```

#### Field Definitions:

*   **`mistakes`**: An array of strings representing active errors.
*   **`confidences`**: Raw scores for all 10 labels (6 Rules, 4 AI).
    *   *Rules*: Binary (0.0 = Pass, 1.0 = Fail).
    *   *AI*: Probabilities (0.0 to 1.0).
*   **`rule_values`**: Geometric measurements for the 6 rule-based labels.
*   **`phase_per_frame`**: The detected `SquatPhase` for every frame in the input sequence.
*   **`joint_heatmap`**: A 2D array of shape `(frames, 36)`.
    *   Indices 0-32: Standard MediaPipe joints.
    *   Indices 33-35: Virtual nodes (Mid-Hip, Mid-Shoulder, Mid-Ear).
    *   Values: Severity (0.0 to 1.0), where 1.0 triggers a full red highlight in the UI.
*   **`phases`**: A summary dictionary. `BOTTOM` contains the specific frame index of the squat apex; others contain frame counts.
