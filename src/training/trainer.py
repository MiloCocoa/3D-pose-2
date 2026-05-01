import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_utils.loader import PoseDataset
from src.model.gcn import MultiLabelGCN
from src.training.metrics import calculate_metrics
from src.config import LABELS, DATA_DIR, VALIDATION_REPORT_PATH, MODEL_SAVE_DIR, MODEL_NAME, LEARNING_RATE, BATCH_SIZE, EPOCHS
import os

def train():
    # Parameters
    data_dir = DATA_DIR
    batch_size = BATCH_SIZE
    num_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    dataset = PoseDataset(data_dir=data_dir, report_path=VALIDATION_REPORT_PATH)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Model, Loss, Optimizer
    model = MultiLabelGCN().to(device)
    # Using BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        all_y_true = []
        all_y_pred = []
        
        for batch in train_loader:
            poses = batch['pose'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(poses)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            all_y_true.append(labels)
            all_y_pred.append(logits)
            
        # Calculate metrics for the epoch
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_metrics(torch.cat(all_y_true), torch.cat(all_y_pred))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Subset Acc: {metrics['subset_accuracy']:.4f}")

    # Save model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}.")

if __name__ == "__main__":
    train()
