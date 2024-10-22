import csv

import networkx as nx

from shortest_path import get_salesmen_stops, create_path

import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split

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

town_list = [
    "dublin",       "antrim",   "craigavon",    "carlow",       "cavan",        "ennis",
    "cork",         "derry",    "letterkenny",  "belfast",      "enniskillen",  "galway",
    "tralee",       "naas",     "kilkenny",     "portlaoise",   "carrick",      "limerick",
    "longford",     "dundalk",  "castlebar",    "navan",        "monaghan",     "tullamore",
    "roscommon",    "sligo",    "clonmel",      "omagh",        "waterford",    "athlone",
    "wexford",      "bray"
]

mapping = {
    "dublin":       0,    "antrim":       1,    "craigavon":    2,    "carlow":       3,    "cavan":        4,    "ennis":        5,
    "cork":         6,    "derry":        7,    "letterkenny":  8,    "belfast":      9,    "enniskillen": 10,    "galway":      11,
    "tralee":      12,    "naas":        13,    "kilkenny":    14,    "portlaoise":  15,    "carrick":     16,    "limerick":    17,
    "longford":    18,    "dundalk":     19,    "castlebar":   20,    "navan":       21,    "monaghan":    22,    "tullamore":   23,
    "roscommon":   24,    "sligo":       25,    "clonmel":     26,    "omagh":       27,    "waterford":   28,    "athlone":     29,
    "wexford":     30,    "bray":        31
}

def get_full_graph():
    return full_graph

# max_arrival  
# scenario    1      2      3  
# MIN_SIM     687    724    484
# MAX_SIM    7732   7392   8378

# sum_arrival
# scenario    1      2      3
# MIN_SIM     687    724    484
# MAX_SIM   16288  15604  20097
def get_MINMAX_sims(is_max_arrival, scenario):
    if is_max_arrival:
        if scenario == 1:
            return 687.0, 7732.0
        elif scenario == 2:
            return 724.0, 7392.0
        else:
            return 484.0, 8378.0
    else:
        if scenario == 1:
            return 687.0, 16288.0
        elif scenario == 2:
            return 724.0, 15604.0
        else:
            return 484.0, 20097.0

def normalize(n, MIN_SIM, MAX_SIM):
    return ((n - MIN_SIM) / (MAX_SIM - MIN_SIM))

def denormalize(n, MIN_SIM, MAX_SIM):
    return (n * (MAX_SIM - MIN_SIM) + MIN_SIM)
    
def get_graph_embeddings1(directory, filename):

    embedding = [0] * (32*32)    
    salesmen = get_salesmen_stops(directory + "/" + filename)

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
                w = full_graph[u][v]['weight'] / (5 * 135.0) # normalization
                
                u_id = mapping[u]
                v_id = mapping[v]
                
                edge_idx = v_id * 32 + u_id
                embedding[edge_idx] += w
                
    return embedding

def get_graph_embeddings2(directory, filename):

    visited = []
    
    salesmen = get_salesmen_stops(directory + "/" + filename)
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
                w = full_graph[u][v]['weight'] / 135.0 # normalization
                
                visited.append((u,v,w))

    embedding = [0] * (2 * len(edges))
    for (u, v, w) in visited:
        idx = -1
        try:
            idx = edges.index((u,v,w))
        except:
            idx = edges.index((v,u,w)) + len(edges)
        embedding[idx] += w
    
    return embedding

def salesmen_to_graph(salesmen):
    # node_features: Tensor of shape [num_nodes, num_node_features]    
    nodes = []
    for i in range(32):
        node = [0] * 32
        node[i] = 1
        nodes.append(node)
    node_features = torch.tensor(nodes, dtype=torch.float)
    assert node_features.shape == torch.Size([32, 32])
    
    sources = []
    targets = []
    weights = []
    attributes = []

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
                w = full_graph[u][v]['weight'] / 135.0 # normalization
                
                u_id = mapping[u]
                v_id = mapping[v]
                
                sources.append(u_id)
                targets.append(v_id)
                attributes.append([w])
                weights.append(w)

    # edge_index: Tensor of shape [2, num_edges] (defines the connections between cities)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    num_edges = len(sources)
    assert num_edges == len(targets)
    assert edge_index.shape == torch.Size([2, num_edges])
    
    # edge_attr: Tensor of shape [num_edges, num_edge_features] (e.g., distance, road conditions)   
    edge_attr = torch.tensor(attributes, dtype=torch.float)
    assert edge_attr.shape == torch.Size([num_edges, 1])
    
    edge_weight = torch.tensor(weights, dtype=torch.float)
    assert edge_weight.shape == torch.Size([num_edges])

    # Create a Data object representing the directed graph
    graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
    return graph


def get_graph_data(directory, filename):   
    salesmen = get_salesmen_stops(directory + "/" + filename)
    return salesmen_to_graph(salesmen)

def build_file(i, j):
    f = ''
    if i < 10:
        f += '0'
    f += str(i) + '-'
    if j < 100:
        f += '0'
    if j < 10:
        f += '0'
    f += str(j) + '.rou.xml'
    return f

def get_graph_list(directory="salesmenroutes"):
    graphs = []
    for i in range(1, 11):
        for j in range(1000):
            f = build_file(i, j)
            graphs.append(get_graph_data(directory, f))
    print(str(len(graphs)) + " graphs read")
    return graphs
 
def get_embedding_list(directory="salesmenroutes"):
    embeddings = []
    for i in range(1, 11):
        for j in range(1000):
            f = build_file(i, j)
            embeddings.append(get_graph_embeddings2(directory, f))
    print(str(len(embeddings)) + " graphs read")
    return embeddings    

def get_sims(max_arrival, scenario):
    MIN_SIM, MAX_SIM = get_MINMAX_sims(max_arrival, scenario)
    csv_path = "results/results_scenario" + str(scenario) + ".csv"
        
    return get_simulation_results(csv_path, MIN_SIM, MAX_SIM, max_arrival)

def get_simulation_results(csv_path, MIN_SIM, MAX_SIM, max_arrival=True):
    sims = []
    r = 2
    if max_arrival:
        r = 1
    
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            sims.append(normalize(float(row[r]), MIN_SIM, MAX_SIM))
                
    assert min(sims) >= 0.00 and min(sims) <  0.01
    assert max(sims) >  0.99 and max(sims) <= 1.01
    return sims

def split_dataset(graph_list, y_list, train_size=0.6, val_size=0.2, test_size=0.2):    
    full_dataset = GraphDataset(graph_list, y_list)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    return train_set, val_set, test_set

def split_embedding_dataset(graph_list, y_list, train_size=0.6, val_size=0.2, test_size=0.2):
    dataset_size = len(graph_list)
    train, val, test = random_split(range(dataset_size), [train_size, val_size, test_size])
    
    train_x = []
    train_y = []
    for i in train:
        train_x.append(graph_list[i])
        train_y.append(y_list[i])
    
    val_x = []
    val_y = []
    for i in val:
        val_x.append(graph_list[i])
        val_y.append(y_list[i])
    
    test_x = []
    test_y = []
    for i in test:
        test_x.append(graph_list[i])
        test_y.append(y_list[i])
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
    


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
        
    def get_graphs(self):
        return self.graph_list
    
    def get_y(self):
        return self.y_values
