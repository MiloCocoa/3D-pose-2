1. Estimated Data Volume: The Multi-Label "Combinatorial" Challenge

In a multi-label scenario, the model isn't just learning what a "Knee Valgus" looks like; it's learning to distinguish it while "Back Rounding" is also happening.

    The "Power of 200" Rule: For ST-GCNs to generalize, aim for at least 200 positive instances of each label. With 10 labels, if they were perfectly mutually exclusive, you'd need 2,000 reps. Since labels overlap (multi-label), you might get away with 1,200–1,500 unique repetitions, provided they are diverse.

    The Sparsity Problem: If "Label A" only appears 5 times in your 1,500 reps, the model will likely treat it as noise and always predict 0 for that label. You need to ensure a relatively balanced Label Density.

    Validation Split: You’ll need a "Seen Subject" vs. "Unseen Subject" split. If you train on 5 people and test on those same 5 people, your accuracy will be fake. You need at least 5–10 subjects in your test set who the model has never seen before.

2. Key Data Requirements: Quality Over Quantity

Since ST-GCNs operate on coordinates (X,Y,C), the "cleanliness" of those coordinates is more important than the number of videos.
A. Feature Normalization (Critical)

If one person is 6'2" and another is 5'2", or if one stands closer to the camera, the raw coordinates will differ wildly.

    Center the Root: Subtract the "Mid-Hip" or "Pelvis" coordinate from all other joints so the squat starts at (0,0,0).

    Scale Invariance: Divide all coordinates by the torso length (distance between neck and mid-hip). This ensures the model learns angles and relative movement, not absolute pixel distances.

B. Temporal Consistency

    FPS Standardization: If half your data is 30 FPS and half is 60 FPS, the "speed" of the squat doubles in the eyes of the GCN. Pre-process all sequences to a fixed frame count (e.g., 60 frames per squat).

    Confidence Scores (C): Your input should be (X,Y,V), where V is the visibility/confidence score from your pose estimator (like MediaPipe). The ST-GCN uses this to "ignore" joints that are occluded during the descent.

3. Data Augmentation: Synthetic Variety

Since you only have 10 samples, augmentation isn't just a bonus—it's your only path to a working model.
Spatial Augmentations (Graph-based)

    Viewpoint Rotation: Since you have the X and Y coordinates, you can apply a 2D rotation matrix to the entire skeleton. Rotating a squat by ±5–10 degrees simulates a camera that isn't perfectly level.

    Gaussian Noise: Add very slight noise (σ=0.01) to the joint coordinates. This forces the ST-GCN to be robust against "jittery" pose estimation.

    Symmetry Flipping: A squat is (mostly) symmetrical. You can flip the skeleton horizontally (swap left-side joints with right-side joints) to double your data instantly. Note: You must also flip the corresponding labels (e.g., "Left Knee Cave" becomes "Right Knee Cave").

Temporal Augmentations

    Time Warping: Randomly speed up or slow down the sequence by 10–20%. This teaches the model that a "slow error" is the same as a "fast error."

    Window Shifting: If your input window is 60 frames but the squat is 45 frames, randomly pad the beginning or the end with the "standing" pose. This prevents the model from expecting the error to occur at the exact same frame index every time.