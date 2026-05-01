System Architecture Overview

The system takes a continuous stream of 33-node 3D spatial coordinates (x,y,z) from a single depth camera. The pipeline consists of four distinct phases: Data Ingestion & Synthesis, The Master State Machine, The Dual-Headed Forward Pass, and Final Aggregation.
Phase 1: Data Ingestion & Virtual Node Synthesis

Because the standard 33-node topology lacks central torso anchors, the system must mathematically synthesize three "virtual nodes" frame-by-frame before any rules are evaluated.

    Virtual Mid_Hip: The precise spatial midpoint between left_hip (23) and right_hip (24). This serves as the proxy for the pelvis.

    Virtual Mid_Shoulder: The midpoint between left_shoulder (11) and right_shoulder (12). This serves as the proxy for the upper sternum/base of the neck.

    Virtual Mid_Ear: The midpoint between left_ear (7) and right_ear (8). This serves as the proxy for the center of the head.

Phase 2: The Master State Machine (Motion Phase Detection)

To prevent rules from firing at the wrong time (e.g., evaluating depth while the athlete is standing), the system uses the virtual Mid_Hip node's vertical Y-axis velocity (y˙​) to act as a master clock.

    Setup Phase: The Y-position of Mid_Hip is at its maximum, and y˙​ is near zero.

    Descent Phase: The y˙​ becomes continuously negative (moving downward toward the floor).

    Bottom Phase (Apex): The y˙​ crosses zero from negative to positive.

    Ascent Phase: The y˙​ becomes continuously positive.

    Finish Phase: The y˙​ returns to zero, and the Y-position returns to its original maximum.

Phase 3: The Dual-Headed Forward Pass

The data is fed simultaneously into the Rule-Based Head and the ST-GCN Head. Each head has absolute authority over its assigned labels based on the Back Squat Assessment (BSA).
Head A: The Rule-Based Module (Geometric Authority)

This deterministic head calculates angles, vectors, and positional deltas using the X (frontal), Y (vertical), and Z (sagittal) coordinates. It evaluates 6 of the 10 labels. For each rule, a specific error tolerance threshold (Δ) is applied. If a threshold is breached, the label is flagged as an error (1).

    Head Position: The BSA requires that the line of the neck is perpendicular to the ground.

        Logic: Evaluated continuously. The system calculates the vector between the virtual Mid_Shoulder and virtual Mid_Ear. It calculates the angle of this vector relative to the absolute vertical Y-axis. If this angle exceeds a defined ΔHead_Tilt​, an error is flagged.

    Hip Position: The line of the hips must be parallel to the ground in the frontal plane.

        Logic: Evaluated during Descent, Bottom, and Ascent. Looking purely at the frontal X/Y plane, the system compares the vertical Y-coordinates of the left and right hips to detect asymmetrical tilt (ΔTilt​). It also tracks the horizontal X-coordinates of the hips relative to the virtual Mid_Hip to detect lateral shifting (ΔShift​).

    Frontal Knee Position: The lateral aspect of the knee must not cross the medial malleolus (inner ankle).

        Logic: Evaluated during Descent, Bottom, and Ascent. Projecting onto the frontal X plane, the system compares the X-coordinate of the knee to the X-coordinate of the ankle on the same leg. If the knee collapses inward past the ankle by a margin greater than ΔValgus​, a dynamic valgus error is flagged.

    Tibial Progression Angle: The tibias should be parallel to an upright torso.

        Logic: Evaluated during Descent and Bottom. In the sagittal Y/Z plane, the system creates a lower-leg vector (knee to ankle) and an upper-torso vector (virtual Mid_Shoulder to virtual Mid_Hip). If the angular difference between these two vectors exceeds ΔParallel​, an error is flagged.

    Foot Position: The entire foot remains in contact with the ground.

        Logic: Evaluated continuously. The system establishes a baseline ground plane Y-coordinate during the Setup phase using the heel and toe (foot index) nodes. If the Y-coordinate of any heel or toe rises above this established ground plane by more than ΔLift​, a heel-raise or toe-raise error is flagged.

    Depth: The tops of the thighs must be at least parallel to the ground.

        Logic: Triggered exclusively at the Bottom Phase. The system compares the vertical Y-coordinates of the hip nodes to the knee nodes. If the hips do not drop to or below the vertical level of the knees, a depth error is flagged.

Head B: The ST-GCN Module (Spatio-Temporal Authority)

This probabilistic head uses the graph sequence to learn complex, dynamic relationships over time. It bypasses the missing torso nodes by observing how the entire skeletal structure moves in harmony (or disharmony). It outputs confidence probabilities (logits) for 4 of the 10 labels, which are passed through a Sigmoid activation and a 0.5 threshold to become binary classifications (0 or 1).

    Thoracic Position: Requires the chest to be held upward and shoulder blades retracted.

        Logic: The ST-GCN learns to identify the signature forward-rolling kinematic pattern of the shoulders relative to the head and hips that indicates a collapsed upper back.

    Trunk Position: Requires maintaining a slightly lordotic lumbar spine.

        Logic: Instead of trying to mathematically map a spine that isn't there, the ST-GCN identifies the subtle, high-frequency wavering, instability, and overall core collapse associated with losing lumbar tension during the movement.

    Descent Mechanics: Requires utilizing a hip-hinge strategy at a controlled, constant speed.

        Logic: The ST-GCN analyzes the velocity profiles of the knees advancing forward in the Z-plane versus the hips traveling backward in the Z-plane to detect if the athlete is "knee-loading" instead of "hip-hinging," as well as identifying sudden, uncontrolled drops in velocity.

    Ascent Mechanics: Shoulders and hips must rise at the same constant speed, and the descent-to-ascent timing ratio must be at least 2:1.

        Logic: The ST-GCN inherently encodes temporal durations. It learns to flag reps where the hips shoot up faster than the shoulders out of the bottom position (the "Good Morning" squat error) and identifies when the overall eccentric-to-concentric tempo ratio is violated.

Phase 4: Final Aggregation

At the end of a recorded repetition, the deterministic output array from Head A (e.g., [Head=0, Hip=1, Knee=0, Tibia=0, Foot=1, Depth=0]) is concatenated with the thresholded probabilistic array from Head B (e.g., [Thoracic=0, Trunk=1, Descent=0, Ascent=0]).

This yields your final, unified 10-label multi-hot vector, perfectly blending exact geometric checks with advanced pattern recognition.