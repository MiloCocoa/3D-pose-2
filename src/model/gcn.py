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
        
        # Multi-Stream Segment-Aware Architecture
        # Stream A (Posture): Spatial Coordinates (x, y, z, vis) for ALL 100 frames
        self.spatial_backbone = nn.ModuleList()
        self.spatial_backbone.append(GCNBlock(TARGET_FRAMES * 4, hidden_channels, DROPOUT))
        for _ in range(NUM_GCN_BLOCKS - 1):
            self.spatial_backbone.append(GCNBlock(hidden_channels, hidden_channels, DROPOUT))
            
        # Stream B1 (Descent): Biomechanical Features (Trunk, L-Knee, R-Knee) for FIRST 50 frames
        self.descent_backbone = nn.ModuleList()
        self.descent_backbone.append(GCNBlock(50 * 3, hidden_channels, DROPOUT))
        for _ in range(NUM_GCN_BLOCKS - 1):
            self.descent_backbone.append(GCNBlock(hidden_channels, hidden_channels, DROPOUT))
            
        # Stream B2 (Ascent): Biomechanical Features (Trunk, L-Knee, R-Knee) for LAST 50 frames
        self.ascent_backbone = nn.ModuleList()
        self.ascent_backbone.append(GCNBlock(50 * 3, hidden_channels, DROPOUT))
        for _ in range(NUM_GCN_BLOCKS - 1):
            self.ascent_backbone.append(GCNBlock(hidden_channels, hidden_channels, DROPOUT))
            
        self.class_pool = global_mean_pool
        
        # Classifiers
        # Thoracic and Trunk use only Posture (Spatial)
        self.posture_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_channels // 2, 2)
        )
        
        # Descent uses Posture + Descent Biomechanics
        self.descent_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Ascent uses Posture + Ascent Biomechanics
        self.ascent_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x):
        # x: [batch_size, num_joints, frames * coords] -> e.g., [B, 36, 700]
        B, N, F = x.shape
        x_split = x.reshape(B, N, TARGET_FRAMES, INPUT_FEATURES)
        
        # Stream A: Features 0-3 (Spatial: X, Y, Z, Vis) over 100 frames
        spatial_x = x_split[..., :4].reshape(B * N, TARGET_FRAMES * 4)
        spatial_x = torch.nan_to_num(spatial_x)
        
        # Stream B1/B2: Features 4-6 (Biomechanics) split into 50 frames each
        descent_x = x_split[:, :, :50, 4:7].reshape(B * N, 50 * 3)
        descent_x = torch.nan_to_num(descent_x)
        
        ascent_x = x_split[:, :, 50:, 4:7].reshape(B * N, 50 * 3)
        ascent_x = torch.nan_to_num(ascent_x)
        
        edge_index = self.edge_index.to(x.device)
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
        
        # Process Spatial
        out_spatial = spatial_x
        for blk in self.spatial_backbone:
            out_spatial = blk(out_spatial, edge_index)
        pooled_spatial = self.class_pool(out_spatial, batch_vec)
            
        # Process Descent
        out_descent = descent_x
        for blk in self.descent_backbone:
            out_descent = blk(out_descent, edge_index)
        pooled_descent = self.class_pool(out_descent, batch_vec)
            
        # Process Ascent
        out_ascent = ascent_x
        for blk in self.ascent_backbone:
            out_ascent = blk(out_ascent, edge_index)
        pooled_ascent = self.class_pool(out_ascent, batch_vec)
            
        # Classification
        posture_logits = self.posture_classifier(pooled_spatial) # [B, 2] (Thoracic, Trunk)
        
        descent_fused = torch.cat([pooled_spatial, pooled_descent], dim=-1)
        descent_logits = self.descent_classifier(descent_fused) # [B, 1] (Descent)
        
        ascent_fused = torch.cat([pooled_spatial, pooled_ascent], dim=-1)
        ascent_logits = self.ascent_classifier(ascent_fused) # [B, 1] (Ascent)
        
        # Concatenate in order: Thoracic (0), Trunk (1), Descent (2), Ascent (3)
        logits = torch.cat([posture_logits, descent_logits, ascent_logits], dim=-1)
        
        return logits
