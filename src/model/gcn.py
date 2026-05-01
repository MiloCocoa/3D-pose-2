import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from src.config import SKELETON_EDGES, NUM_JOINTS, LABELS, HIDDEN_CHANNELS, NUM_GCN_BLOCKS, DROPOUT, TARGET_FRAMES, INPUT_FEATURES

def create_skeleton_graph():
    """
    Creates the static graph connectivity (edge_index) for the 33-joint skeleton.
    """
    edge_index_list = []
    for edge in SKELETON_EDGES:
        edge_index_list.append([edge[0], edge[1]])
        edge_index_list.append([edge[1], edge[0]])
        
    # Add self-loops
    for i in range(NUM_JOINTS):
         edge_index_list.append([i, i])
        
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return edge_index

class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultiLabelGCN(nn.Module):
    def __init__(self, in_channels=TARGET_FRAMES * INPUT_FEATURES, hidden_channels=HIDDEN_CHANNELS, num_labels=4):
        super(MultiLabelGCN, self).__init__()
        self.edge_index = create_skeleton_graph()
        
        # Backbone
        self.backbone = nn.ModuleList()
        self.backbone.append(GCNBlock(in_channels, hidden_channels, DROPOUT))
        for _ in range(NUM_GCN_BLOCKS - 1):
            self.backbone.append(GCNBlock(hidden_channels, hidden_channels, DROPOUT))
            
        # Classification Head
        self.class_head_gcn = GCNConv(hidden_channels, hidden_channels)
        self.class_pool = global_mean_pool
        self.classifier = nn.Linear(hidden_channels, num_labels)

    def forward(self, x):
        # x: [batch_size, num_joints, frames * coords] -> e.g., [B, 33, 300]
        B, N, F = x.shape
        x = x.reshape(B * N, F)
        
        # Ensure no NaNs enter the network
        x = torch.nan_to_num(x)
        
        edge_index = self.edge_index.to(x.device)
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
        
        # Shared Backbone
        for blk in self.backbone:
            x = blk(x, edge_index)
            
        # Classifier Head
        x = self.class_head_gcn(x, edge_index)
        x_pooled = self.class_pool(x, batch_vec)
        logits = self.classifier(x_pooled)
        
        return logits
