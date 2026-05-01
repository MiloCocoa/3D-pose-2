Phase 1: The "Smoke Test" (Using your 10 samples)

Before recording 1,000 videos, ensure your current pipeline is logically sound.

    Overfit the 10 samples: Run your training script on just these 10 samples for 100+ epochs.

        Goal: Your loss should drop to nearly zero, and accuracy should hit 100% on the training set.

        Why? If an ST-GCN cannot memorize 10 samples, there is a bug in your coordinate normalization or your Adjacency Matrix.

    Inspect the Adjacency Matrix: Ensure your graph edges actually connect the joints. For a squat, the connection between Mid-Hip → Knee → Ankle is the most critical path.

    Verify Multi-Label Output: Ensure your final layer is a Linear layer with 10 outputs and that you are using BCEWithLogitsLoss.

Phase 2: Data Preprocessing & Normalization

GCNs are extremely sensitive to "Global Motion" (the person moving around the frame). You need to strip that away so the model only sees the "Relative Motion" (the squat).

    Root Centering: For every frame, subtract the coordinates of the "Mid-Hip" (or Pelvis) from all other joints. This makes the hip (0,0).

    Unit Scaling: Divide all coordinates by the distance between the Neck and the Mid-Hip. This "standardizes" every human to the same relative size.

    Temporal Padding/Sampling: Decide on a fixed window (e.g., 60 frames).

        If a squat is 40 frames, pad with zeros or repeat the last frame.

        If a squat is 100 frames, downsample it to 60.

Phase 3: The "Pilot" Collection (Target: 100–200 Reps)

Don't jump to 2,000 reps yet. Collect a "Pilot" batch to see which errors are hardest to detect.

    Diversity of Errors: Record yourself or a few friends performing "Pure" errors (only one error at a time) and "Compound" errors (2–3 errors at once).

    Subject Variation: Get at least 5 different body types.

    Angle Variation: Record from 45° (side-front) and 90° (pure profile). Avoid pure front-on views for squats, as depth (hips moving back) is hard for 2D pose estimators to see.

Phase 4: Implementing the Augmentation Pipeline

Since your dataset is small, your Dataset class should apply transformations on the fly during training.

    Horizontal Flip: * Flip the X-coordinates: xnew​=1−xold​.

        Crucial: Swap the joint indices (Left Knee ↔ Right Knee) and the labels (Label "Left Knee Cave" ↔ "Right Knee Cave").

    Temporal Shear: Randomly skip every 2nd frame or double-up frames to simulate different squat speeds.

    Coordinate Jitter: Add a tiny amount of Gaussian noise to the (x,y) values.

Phase 5: Training & Iteration

    Weighting the Loss: In multi-label tasks, some errors (like "Heels Lifting") might be rare. Use a pos_weight in your BCEWithLogitsLoss to tell the model that missing a rare error is a "bigger mistake" than missing a common one.

    Threshold Tuning: After training, don't just use 0.5 as your cutoff for every label. Use a Precision-Recall curve to find the best threshold for each of the 10 errors.

Summary Checklist

    [ ] Now: Overfit your 10 samples to 0 loss.

    [ ] Next Week: Normalize coordinates and implement the "Horizontal Flip" augmentation.

    [ ] Next Month: Aim for 200 high-quality, multi-label annotated repetitions.