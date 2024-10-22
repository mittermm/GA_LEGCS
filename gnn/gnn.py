import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        
        # Node feature input size: 2 -> Hidden size: 32
        self.node_emb = torch.nn.Linear(2, 32)
        
        # Edge feature input size: 1 -> Hidden size: 32
        self.edge_emb = torch.nn.Linear(1, 32)
        
        # Graph convolution layers
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 64)
        
        # Fully connected layer for graph-level output
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Embed node features
        x = self.node_emb(x)
        x = F.relu(x)
        
        # Embed edge features
        edge_attr = self.edge_emb(edge_attr)
        edge_attr = F.relu(edge_attr)

        # Apply graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pooling (aggregate node features to a graph-level representation)
        x = global_mean_pool(x, data.batch)  # Assumes we are batching graphs
        
        # Fully connected layers to predict graph-level output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output a single scalar per graph

        return x
