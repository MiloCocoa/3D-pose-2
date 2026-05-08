
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from src.data_utils.loader import PoseDataset
from src.model.gcn import MultiLabelGCN
from src.config import DATA_DIR, VALIDATION_REPORT_PATH, LABELS
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/multi_label_gcn_v2.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Load Dataset (using the same loader as training for consistency)
    test_dir = os.path.join(DATA_DIR, "test")
    dataset = PoseDataset(data_dir=test_dir, report_path=None)
    
    # Fallback if the report filtered out all files in the manually split folder
    if len(dataset) == 0:
        print(f"Warning: Dataset is empty after filtering. Falling back to loading all files in {test_dir} directly.")
        dataset = PoseDataset(data_dir=test_dir, report_path=None)

    # Use a small subset or the whole thing if it's the test set
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load Model
    model = MultiLabelGCN(num_labels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_y_true = []
    all_y_pred = []

    print(f"Evaluating model on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in eval_loader:
            poses = batch['pose'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(poses)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_y_true.append(labels.cpu().numpy()[0])
            all_y_pred.append(preds[0])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # AI labels are Thoracic, Trunk, Descent, Ascent (indices 6, 7, 8, 9 in the global LABELS list)
    ai_label_names = ["Thoracic", "Trunk", "Descent", "Ascent"]

    print("\n" + "="*60)
    print(f"{'Label':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)

    for i, name in enumerate(ai_label_names):
        y_true = all_y_true[:, i]
        y_pred = all_y_pred[:, i]
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        print(f"{name:<15} | {acc:<10.4f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

    # Overall Subset Accuracy (all labels correct)
    subset_acc = np.all(all_y_true == all_y_pred, axis=1).mean()
    print("-" * 60)
    print(f"{'OVERALL SUBSET ACCURACY:':<40} {subset_acc:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluate()
