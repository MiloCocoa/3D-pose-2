# 3D Barbell Squat Form Correction

This project uses a **Dual-Head** architecture to provide real-time (and offline) feedback on barbell squat form using 3D pose data from MediaPipe.

## Architecture

The system splits the 10 analysis labels into two specialized heads:

1.  **Rule-Based Head (Geometric/Safety)**:
    - Tracks 6 deterministic labels using geometric thresholds.
    - Labels: `Head`, `Hip`, `Frontal Knee` (Valgus), `Tibial Angle` (Parallelism), `Foot` (Lift), `Depth`.
    - Leverages **Virtual Node Synthesis** (Mid-Hip, Mid-Shoulder, Mid-Ear) to improve accuracy.

2.  **AI Head (ST-GCN - Postural/Mechanics)**:
    - Uses a Spatio-Temporal Graph Convolutional Network specialized for 4 complex postural labels.
    - Labels: `Thoracic` (Extension), `Trunk` (Forward Lean), `Descent` (Mechanics), `Ascent` (Mechanics).
    - Targeted training on the last 4 indices of the label set.

## Key Features

- **Virtual Node Synthesis**: Synthesizes a virtual spine proxy (Mid-Hip to Mid-Shoulder) for better torso lean analysis.
- **Squat State Machine**: Automatically detects phases of motion (`SETUP`, `DESCENT`, `BOTTOM`, `ASCENT`, `FINISH`).
- **3D React Frontend**: Provides an interactive 3D skeleton visualizer with mistake-specific joint highlighting.
- **Mistake Categorization**: Groups feedback into "Geometric / Safety" and "Postural / Mechanics".

## Setup & Usage

### Backend
1.  Ensure you have Python 3.10+ and the required packages installed from `requirements.txt`.
2.  Run the API:
    ```bash
    python -m src.api.main
    ```

### Frontend
1.  Navigate to `frontend/`.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the dev server:
    ```bash
    npm run dev
    ```

### Analysis
Upload a JSON file containing a MediaPipe `pose_sequence` to the web interface to see real-time 3D feedback and analysis.
