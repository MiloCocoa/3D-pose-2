import torch
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

def calculate_metrics(y_true, y_pred_logits, threshold=0.5):
    # y_true: (batch, num_labels)
    # y_pred_logits: (batch, num_labels)
    
    # Convert logits to probabilities and then to binary predictions
    y_pred_probs = torch.sigmoid(y_pred_logits).cpu().detach().numpy()
    y_pred = (y_pred_probs > threshold).astype(int)
    y_true = y_true.cpu().detach().numpy()
    
    # Hamming Loss: The fraction of labels that are incorrectly predicted.
    h_loss = hamming_loss(y_true, y_pred)
    
    # F1 Score (Micro): Good for multi-label classification
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # F1 Score (Macro): Average F1 across labels
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Subset Accuracy (Exact Match Ratio): Percentage of samples where all labels are correctly predicted.
    subset_acc = accuracy_score(y_true, y_pred)
    
    return {
        'hamming_loss': h_loss,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'subset_accuracy': subset_acc
    }
