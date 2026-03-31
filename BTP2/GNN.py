import torch
from torch_geometric.nn import radius_graph, GCNConv
from torch_geometric.data import Data, Batch

class SwarmGNN(torch.nn.Module):
    def __init__(self, feature_dim):
        super(SwarmGNN, self).__init__()
        self.conv1 = GCNConv(feature_dim, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, x, pos, r):
        """
        x: Node features (e.g. battery, sensor) [N, feature_dim]
        pos: 2D/3D Coordinates [N, 2 or 3]
        r: Communication/Sensing radius
        """
        # 1. Dynamically find neighbors within radius 'r'
        # This returns a new edge_index based on current positions
        edge_index = radius_graph(pos, r=r, loop=False)
        
        # 2. Calculate distances for edge weights
        # Higher weight for closer bots
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).pow(2).sum(-1).sqrt()
        edge_weight = 1.0 / (dist + 1e-6)

        # 3. Message Passing
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        
        return x

# Example: Time Step 1 (3 Robots)
pos_t1 = torch.tensor([[0,0], [0,1], [5,5]], dtype=torch.float) # Bot 3 is far away
feat_t1 = torch.randn(3, 4)

# Example: Time Step 2 (5 Robots - 2 joined the swarm)
pos_t2 = torch.tensor([[0,0], [0,1], [1,1], [0.5, 0.5], [2,2]], dtype=torch.float)
feat_t2 = torch.randn(5, 4)

model = SwarmGNN(feature_dim=4)
out_t1 = model(feat_t1, pos_t1, r=2.0)
out_t2 = model(feat_t2, pos_t2, r=2.0)

print(f"Output for 3 bots: {out_t1.shape}")
print(f"Output for 5 bots: {out_t2.shape}")