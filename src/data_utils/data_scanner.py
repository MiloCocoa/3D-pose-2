import os
import json
import numpy as np
import pandas as pd
from src.config import NUM_JOINTS, DATA_DIR, VALIDATION_REPORT_PATH, MAX_GAP_SIZE

def scan_dataset(data_dir=DATA_DIR, output_report=VALIDATION_REPORT_PATH):
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    report = []
    
    print(f"Scanning {len(files)} files in {data_dir}...")
    
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        status = "PASS"
        issues = []
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # 1. Basic Structure Checks
            if 'metadata' not in data or 'label' not in data['metadata']:
                status = "FAIL"
                issues.append("Missing labels in metadata")
            elif len(data['metadata']['label']) != 10:
                status = "FAIL"
                issues.append(f"Expected 10 labels, found {len(data['metadata']['label'])}")
            
            if 'pose_sequence' not in data or len(data['pose_sequence']) == 0:
                status = "FAIL"
                issues.append("Empty pose sequence")
            else:
                # 2. Detailed Data Analysis
                pose_seq_list = []
                for frame in data['pose_sequence']:
                    frame_coords = []
                    for joint in frame:
                        # Convert None to np.nan explicitly
                        x = joint.get('x_3d_meters')
                        y = joint.get('y_3d_meters')
                        z = joint.get('z_3d_meters')
                        frame_coords.append([
                            float(x) if x is not None else np.nan,
                            float(y) if y is not None else np.nan,
                            float(z) if z is not None else np.nan
                        ])
                    pose_seq_list.append(frame_coords)
                
                pose_seq = np.array(pose_seq_list) # (frames, joints, 3)

                # Check for large gaps
                # Any joint coordinate that is NaN
                nan_mask = np.isnan(pose_seq).any(axis=2) # (frames, joints)
                
                for j in range(NUM_JOINTS):
                    joint_nan_mask = nan_mask[:, j]
                    if np.any(joint_nan_mask):
                        # Find max contiguous NaNs
                        max_gap = 0
                        current_gap = 0
                        for val in joint_nan_mask:
                            if val:
                                current_gap += 1
                                max_gap = max(max_gap, current_gap)
                            else:
                                current_gap = 0
                        
                        if max_gap > MAX_GAP_SIZE:
                            status = "FAIL"
                            issues.append(f"Joint {j} has gap of {max_gap} (max allowed {MAX_GAP_SIZE})")
                        elif max_gap > 0:
                            issues.append(f"FIXABLE: Joint {j} gap {max_gap}")

                # Check for outliers (ignoring NaNs)
                valid_coords = pose_seq[~np.isnan(pose_seq)]
                if len(valid_coords) > 0 and np.any(np.abs(valid_coords) > 10.0):
                    status = "FAIL"
                    issues.append("Coordinate outliers > 10m")

        except Exception as e:
            status = "FAIL"
            issues.append(f"Load Error: {str(e)}")
            
        report.append({
            "file": fname,
            "status": status,
            "issues": " | ".join(issues)
        })
        
    # Save Report
    df = pd.DataFrame(report)
    df.to_csv(output_report, index=False)
    
    # Summary
    failed = df[df['status'] == "FAIL"]
    print(f"\nScan Complete!")
    print(f"Total Files: {len(df)}")
    print(f"Passed (Clean or Fixable): {len(df) - len(failed)}")
    print(f"Failed (Unrecoverable): {len(failed)}")
    print(f"Report saved to: {output_report}")
    
    if len(failed) > 0:
        print("\nFirst 5 failed files:")
        print(failed[['file', 'issues']].head(5))

if __name__ == "__main__":
    scan_dataset()
