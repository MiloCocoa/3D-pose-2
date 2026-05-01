# Multi-Label 3D Pose Correction: Implementation Plan

This document serves as a comprehensive guide for building a new repository dedicated to multi-label 3D barbell squat form correction. Use these steps as directives for Gemini CLI to scaffold and implement the project from scratch.

## Project Overview
- **Goal:** Classify multiple simultaneous squat mistakes and generate corrected 3D poses.
- **Task:** Multi-label classification (outputs: list of detected faults) + Sequence-to-Sequence regression (output: corrected 3D coordinates).
- **Stack:** PyTorch, PyTorch Geometric, FastAPI, React (Vite).

---

## Phase 1: Environment & Scaffolding
**Goal:** Initialize the project structure and dependencies.

1. **Initialize Project:** Create the following directory structure:
   ```text
   /
   ├── data/             # Dataset storage
   ├── src/
   │   ├── api/          # FastAPI backend
   │   ├── model/        # GCN architecture
   │   ├── data_utils/   # Data loaders and preprocessing
   │   └── training/     # Training and evaluation logic
   └── frontend/         # React (Vite) application
   ```
2. **Setup Requirements:** Create a `requirements.txt` including `torch`, `torch-geometric`, `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, and `numpy`.
3. **Configuration:** Create `src/config.py` to define:
   - `LABELS`: List of possible mistakes (e.g., ["butt_wink", "bar_tilt", "knees_in"]).
   - `THRESHOLD`: Confidence value for active labels (default: 0.5).
   - `JOINT_MAP`: Indexing for the 3D skeleton.

---

## Phase 2: Data Layer (Multi-Label)
**Goal:** Implement data handling for multi-hot labels.

1. **Multi-Label Loader:** Create `src/data_utils/loader.py`.
   - Implement a PyTorch `Dataset` that expects labels as multi-hot vectors (e.g., `[1.0, 0.0, 1.0]`).
   - Include sequence resampling to ensure all input clips have a fixed number of frames.
2. **Validation Script:** Create `src/data_utils/validate.py` to check that the dataset format matches the expected tensor shapes: `(batch, frames, joints, 3)` for poses and `(batch, num_labels)` for targets.

---

## Phase 3: Model Architecture
**Goal:** Build the dual-branch GCN.

1. **GCN Backbone:** Create `src/model/gcn.py` using PyTorch Geometric.
   - Implement a Graph Convolutional Network that processes spatial (skeleton edges) and temporal (frame sequences) data.
2. **Classification Branch:** Ensure the final layer outputs `len(LABELS)` units. 
   - *Note:* Do not use Softmax; use a raw linear output (logits) to be paired with `BCEWithLogitsLoss`.
3. **Correction Branch:** Implement a decoder branch that outputs a sequence of the same shape as the input, representing the "ideal" pose.

---

## Phase 4: Training & Multi-Label Metrics
**Goal:** Implement logic for simultaneous mistake detection.

1. **Loss Function:** In `src/training/trainer.py`, implement a combined loss:
   - `Classification Loss`: `torch.nn.BCEWithLogitsLoss` (handles multiple active labels).
   - `Correction Loss`: `MSELoss` for coordinate regression.
2. **Metrics:** Create `src/training/metrics.py` to calculate:
   - **Hamming Loss:** Fraction of wrong labels.
   - **F1-Score (Macro/Micro):** Performance across multiple categories.
   - **Subset Accuracy:** Percentage of samples where all labels are correctly predicted.

---

## Phase 5: Backend API
**Goal:** Create an inference engine.

1. **Inference Script:** Create `src/inference.py` to load a trained model and convert raw coordinates into a list of strings (active mistakes) based on the `THRESHOLD`.
2. **FastAPI App:** Create `src/api/main.py` with an `/analyze` endpoint.
   - Input: JSON sequence of 3D coordinates.
   - Output: 
     ```json
     {
       "mistakes": ["butt_wink", "knees_in"],
       "confidences": {"butt_wink": 0.92, "knees_in": 0.75, "bar_tilt": 0.04},
       "corrected_pose": [...]
     }
     ```

---

## Phase 6: Frontend Visualization
**Goal:** Build a UI that displays multiple results.

1. **React Scaffold:** Initialize a Vite-React app in `frontend/`.
2. **Multi-Label Display:** Modify the results component to:
   - Display a list of "Detected Faults" as tags or alerts.
   - Show a "Health Bar" or confidence score for every tracked mistake.
3. **3D Visualizer:** Implement a Three.js or simple Canvas-based skeleton renderer to toggle between "Original" and "Corrected" poses.
