# Integrated Scaling Plan (2026-03-15)

This document consolidates the strategies for scaling the Multi-Label 3D Squat GCN from a 10-sample prototype to a 1,500+ rep production model.

## Phase 1: The Smoke Test (Validation)
**Goal:** Prove the ST-GCN can memorize the existing 10 samples.
- **Overfit Strategy:** Train for 500+ epochs with `Batch Size = 10` and `Dropout = 0`.
- **Structural Check:** Verify the Adjacency Matrix connects the critical Mid-Hip → Knee → Ankle path.
- **Loss:** Drive `BCEWithLogitsLoss` to near zero.

## Phase 2: Advanced Preprocessing & Normalization
**Goal:** Strip global motion and standardize body types.
- **Root Centering:** Subtract Mid-Hip (midpoint of joints 23, 24) from all coordinates.
- **Torso Scaling:** Divide by the distance between the "Virtual Neck" (mid-shoulders 11, 12) and "Mid-Hip."
- **Feature Depth:** Include MediaPipe `visibility` (Confidence) as a 4th feature (X, Y, Z, V).
- **Temporal Window:** Standardize to 100 frames with linear interpolation.

## Phase 3: Augmentation Pipeline
**Goal:** Synthetic variety to bridge the data gap.
- **Horizontal Flip:**
    - Flip X-coordinates ($x_{new} = -x_{old}$ after centering).
    - Swap joint indices using `SYMMETRY_MAP` (e.g., Left Knee ↔ Right Knee).
    - Toggle labels if they are side-specific.
- **Coordinate Jitter:** Add Gaussian noise ($\sigma=0.01$) to coordinates.
- **Temporal Shear:** Simulate different squat speeds by random frame skipping/repeating.

## Phase 4: Pilot Collection (100–200 Reps)
- **Diversity:** Mix "Pure" and "Compound" errors.
- **Subject Split:** Reserve at least one subject ID exclusively for the test set to verify generalization.

## Phase 5: Metrics & Tuning
- **Weighted Loss:** Use `pos_weight` in `BCEWithLogitsLoss` for rare faults.
- **Threshold Tuning:** Optimize the 0.5 cutoff per label using Precision-Recall curves.
