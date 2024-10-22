import os
import networkx as nx
from shortest_path import get_salesmen_stops, create_path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from sklearn.metrics import mean_absolute_error, r2_score
import csv

import numpy as np

full_graph = nx.Graph()
edges = [
    ("limerick", "tralee", 80), ("limerick", "cork", 87), ("tralee", "cork", 97),
    ("limerick", "ennis", 35), ("limerick", "galway", 75), ("ennis", "galway", 55),
    ("ennis", "athlone", 85), ("limerick", "athlone", 97), ("galway", "athlone", 57),
    ("limerick", "clonmel", 67), ("tralee", "clonmel", 135), ("cork", "clonmel", 70),
    ("clonmel", "waterford", 50), ("cork", "waterford", 105), ("limerick", "kilkenny", 100),
    ("clonmel", "kilkenny", 45), ("waterford", "wexford", 50), ("kilkenny", "waterford", 40),
    ("kilkenny", "wexford", 70), ("limerick", "portlaoise", 75), ("portlaoise", "carlow", 35),
    ("carlow", "kilkenny", 27), ("portlaoise", "kilkenny", 45), ("limerick", "tullamore", 92),
    ("athlone", "tullamore", 34), ("wexford", "carlow", 62), ("tullamore", "portlaoise", 30),
    ("wexford", "bray", 85), ("carlow", "naas", 40), ("portlaoise", "naas", 40),
    ("tullamore", "naas", 60), ("naas", "dublin", 37), ("dublin", "bray", 40),
    ("naas", "bray", 35), ("tullamore", "dublin", 82), ("athlone", "dublin", 92),
    ("galway", "kilkenny", 145), ("dublin", "navan", 50), ("dublin", "dundalk", 62),
    ("navan", "dundalk", 45), ("navan", "athlone", 82), ("navan", "tullamore", 70),
    ("naas", "navan", 55), ("galway", "roscommon", 72), ("athlone", "roscommon", 26),
    ("longford", "roscommon", 28), ("longford", "athlone", 42), ("athlone", "cavan", 75),
    ("longford", "cavan", 47), ("longford", "dundalk", 110), ("longford", "navan", 70),
    ("cavan", "navan", 55), ("cavan", "dundalk", 77), ("naas", "longford", 87),
    ("tullamore", "cavan", 85), ("dundalk", "navan", 120), ("galway", "castlebar", 67),
    ("roscommon", "castlebar", 70), ("longford", "castlebar", 82), ("carrick", "castlebar", 65),
    ("sligo", "castlebar", 65), ("carrick", "roscommon", 35), ("carrick", "longford", 28),
    ("carrick", "cavan", 65), ("carrick", "sligo", 40), ("cork", "portlaoise", 100),
    ("sligo", "enniskillen", 60), ("enniskillen", "carrick", 60), ("enniskillen", "cavan", 45),
    ("monaghan", "cavan", 45), ("enniskillen", "monaghan", 50), ("carrick", "monaghan", 90),
    ("sligo", "cavan", 85), ("enniskillen", "omagh", 40), ("sligo", "omagh", 85),
    ("omagh", "monaghan", 45), ("monaghan", "dundalk", 45), ("letterkenny", "sligo", 90),
    ("letterkenny", "omagh", 50), ("enniskillen", "letterkenny", 75), ("letterkenny", "derry", 35),
    ("derry", "omagh", 45), ("derry", "antrim", 62), ("omagh", "antrim", 72),
    ("antrim", "craigavon", 40), ("craigavon", "monaghan", 45), ("craigavon", "enniskillen", 67),
    ("craigavon", "derry", 82), ("craigavon", "omagh", 50), ("dundalk", "omagh", 80),
    ("dundalk", "craigavon", 45), ("dundalk", "belfast", 57), ("antrim", "belfast", 24),
    ("craigavon", "belfast", 30)
    # ("dublin", "longford", 92)
]
full_graph.add_weighted_edges_from(edges)

mapping = {
    "dublin":       0,    "antrim":       1,    "craigavon":    2,    "carlow":       3,    "cavan":        4,    "ennis":        5,
    "cork":         6,    "derry":        7,    "letterkenny":  8,    "belfast":      9,    "enniskillen": 10,    "galway":      11,
    "tralee":      12,    "naas":        13,    "kilkenny":    14,    "portlaoise":  15,    "carrick":     16,    "limerick":    17,
    "longford":    18,    "dundalk":     19,    "castlebar":   20,    "navan":       21,    "monaghan":    22,    "tullamore":   23,
    "roscommon":   24,    "sligo":       25,    "clonmel":     26,    "omagh":       27,    "waterford":   28,    "athlone":     29,
    "wexford":     30,    "bray":        31
}

directory = "salesmenroutes"
csv_path = "results/results_scenario2.csv"

#configs = []
#for f in files:
#    salesmen = get_salesmen_stops(directory + "/" + f)
#    path = create_path(full_graph, salesmen)
#    configs.append(path)
#    print(f + " " + str(len(path.edges)))
#
#print(len(configs))

MIN_SIM =  724.0
MAX_SIM = 7392.0

def normalize(n):
    return ((n - MIN_SIM) / (MAX_SIM - MIN_SIM))

def denormalize(n):
    return (n * (MAX_SIM - MIN_SIM) + MIN_SIM)

def get_graph_data(filename):
    edge_list = []
    edge_weight_list = []

    salesmen = get_salesmen_stops(directory + "/" + filename)
    #print(salesmen)
    for towns in salesmen:
        end_node = "dublin"
        towns.append("dublin")
        for j in range(len(towns)):
            start_node = end_node
            end_node = towns[j]
            
            shortest_path = nx.shortest_path(full_graph, source=start_node, target=end_node, weight='weight')
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i + 1]
                weight = full_graph[u][v]['weight'] # / 135.0 # normalization
                
                u_id = mapping[u]
                v_id = mapping[v]
                
                edge_list.append([u_id, v_id])
                edge_weight_list.append(weight)
                
    edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weight_list, dtype=torch.float)
        
    num_nodes = edges.max().item() + 1  # number of nodes              
    nodes_list = []
    for i in range(num_nodes):
        node = [0] * num_nodes
        node[i] = 1
        nodes_list.append(node)
    
    # nodes = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)
    nodes = torch.tensor(nodes_list, dtype=torch.float)
    
    # No explicit node features, so we use a learnable embedding
    #node_embedding_dim = 4  # you can choose this dimension
    #x = torch.nn.Embedding(num_nodes, node_embedding_dim)  # Learnable node embeddings

    # Create the Data object
    data = Data(x = nodes, edge_index=edges, edge_attr=edge_weights)
    return data

def create_graph_list():
    graphs = []
    files = os.listdir(directory)
    for f in files:
        graphs.append(get_graph_data(f))
    print(str(len(graphs)) + " graphs read")
    return graphs

def get_simulation_results():
    sims = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            sims.append(normalize(float(row[1])))
    return sims

def train(model, train_loader, val_loader):
    criterion = torch.nn.L1Loss() # torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr = 0.01

    num_epochs = 100 # 100
    for epoch in range(num_epochs):
        model.train()
        for data_i in train_loader:
            optimizer.zero_grad()
            out = model(data_i)

            loss = criterion(out.view(-1), data_i.y)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data_i in val_loader:
                out = model(data_i)
                val_loss += criterion(out.view(-1), data_i.y).item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}')

    return model

def make_predictions(model, testing_dataset):
    preds = []
    for i_data in testing_dataset:
        model.eval()
        with torch.no_grad():
            predicted_output = model(i_data)

        pred = predicted_output.item()
        preds.append(pred)

    return preds

class TrafficGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(TrafficGNN, self).__init__()
        self.conv1 = GCNConv(32, hidden_channels)  # First GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # Second GCN layer
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=1)  # First GAT layer
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)  # Second GAT layer
        
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)  # Fully connected layer 1
        self.fc2 = torch.nn.Linear(hidden_channels // 2, 1)  # Output layer
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        

        # Apply GCN layers with edge weights
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Apply GAT layers
        x = self.gat1(x, edge_index)  # Edge weights can influence attention
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        # Pooling: Aggregate node features into a graph-level feature
        x = global_mean_pool(x, data.batch)  # Using mean pooling
        
        # Apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x.squeeze()

class GraphDataset(Dataset):
    def __init__(self, graph_list, y_values):
        self.graph_list = graph_list
        self.y_values = y_values

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        y_value = self.y_values[idx]
        data = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=torch.tensor(y_value, dtype=torch.float))
        return data

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self, idx):
        return self.__len__()
        
gl = create_graph_list()
sims = get_simulation_results()
full_dataset = GraphDataset(gl, sims)

graph = full_dataset.__getitem__(1)
print(graph)
print("number of graphs:\t\t",len(full_dataset))
print("number of node features:\t",full_dataset.num_node_features)
print("number of edge features:\t",full_dataset.num_edge_features)


train_gl = []
train_sims = []
for i in range(10):
    for j in range(700):
        train_gl.append(gl[i*1000 + j])
        train_sims.append(sims[i*1000 + j])
training_dataset = GraphDataset(train_gl, train_sims)

test_gl = []
test_sims = []
for i in range(10):
    for j in range(700, 1000):
        test_gl.append(gl[i*1000 + j])
        test_sims.append(sims[i*1000 + j])
testing_dataset = GraphDataset(test_gl, test_sims)


training_data = 7000
train_size = int(0.8 * training_data)
val_size = training_data - train_size
train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Example data preparation
node_features = 32  # Only the node indices
edge_features = 1  # Initially, just the estimated travel time
hidden_channels = 64

model = TrafficGNN(hidden_channels)
model = train(model, train_loader, val_loader)

preds = make_predictions(model, testing_dataset)
mae = mean_absolute_error(test_sims, preds)
print(preds)
print("MAE = ", mae)
r2 = r2_score(test_sims, preds)
print("r  = ", r2)


