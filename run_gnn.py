from gnn.gnn_utils import get_graph_list, get_sims, split_dataset
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import Data, DataLoader
from gnn.gnn import GNN
from gnn.gnn_utils import train, evaluate
import torch
import torch.nn.functional as F


graph_list = get_graph_list()
sims_list = get_sims()

if len(graph_list) != len(sims_list):
    minimum = min(len(graph_list), len(sims_list))
    graph_list = graph_list[:minimum]
    sims_list = sims_list[:minimum]

assert len(graph_list) == len(sims_list)

train_dataset, val_dataset, test_dataset = split_dataset(graph_list, sims_list)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model, optimizer, and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the datasets (assuming `train_dataset`, `val_dataset`, `test_dataset`)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training loop with validation MAE and R2 after each epoch
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer)
    val_loss, val_mae, val_r2 = evaluate(model, val_loader)
    
    print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')

# Final evaluation on the test dataset (after training is complete)
test_loss, test_mae, test_r2 = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')

# save model
# torch.save(model.state_dict(), 'gnn/gnn_model.pth')

# load model
# model = GNN()
# model.load_state_dict(torch.load('gnn/gnn_model.pth'))
# model.eval()
