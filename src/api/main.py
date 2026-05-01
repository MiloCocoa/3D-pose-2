from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from src.inference import InferenceEngine
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
model_path = os.path.join("models", "multi_label_gcn.pth")
engine = InferenceEngine(model_path)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Multi-Label 3D Pose Correction API is running."}

@app.post("/analyze")
async def analyze_pose(request: Request):
    try:
        data = await request.json()
        pose_sequence = data.get("pose_sequence")
        
        if not pose_sequence:
            raise HTTPException(status_code=400, detail="Missing pose_sequence")
            
        # Manually extract just the 4 fields we need
        # This is robust against extra fields, missing optional fields, or schema mismatches
        # Manually extract just the 4 fields we need
        processed_sequence = []
        for frame in pose_sequence:
            # Sort joints by index to match config expectations
            try:
                sorted_joints = sorted(frame, key=lambda x: x.get("index", 0))
                frame_data = []
                for j in sorted_joints:
                    # Handle None/null values explicitly
                    def safe_float(val, default=0.0):
                        return float(val) if val is not None else default

                    frame_data.append([
                        safe_float(j.get("x_3d_meters")),
                        safe_float(j.get("y_3d_meters")),
                        safe_float(j.get("z_3d_meters")),
                        safe_float(j.get("visibility"), 1.0)
                    ])
                processed_sequence.append(frame_data)
            except Exception as inner_e:
                print(f"Frame parsing error: {inner_e}")
                # Don't skip the frame, add a dummy one to keep timing consistent
                # 33 joints (original) or 36? The input is 33.
                if processed_sequence:
                    processed_sequence.append(processed_sequence[-1]) # Repeat last frame
                continue

        if not processed_sequence:
            raise HTTPException(status_code=400, detail="Could not parse pose_sequence")
            
        result = engine.predict(processed_sequence)
        # Ensure result contains mistakes, confidences, rule_values, and phase_per_frame
        return result
        
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
