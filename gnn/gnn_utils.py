import csv
from sklearn.preprocessing import normalize
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score


def normalize(n, MIN_SIM, MAX_SIM):
    return ((n - MIN_SIM) / (MAX_SIM - MIN_SIM))

def normalize_list(l):
    nl = []
    minimum = min(l)
    maximum = max(l)
    for i in l:
        nl.append(normalize(i, minimum, maximum))
    return nl, minimum, maximum

def get_town_index(town):
    i = 0
    for (t, x, y) in town_xy_list:
        if t == town:
            return i
        i += 1
    return -1

town_xy_list = [
		("dublin", 0.9295436349079264, 0.4681783045606802), 
		("antrim", 0.9151321056845476, 0.8989951043545478), 
		("craigavon", 0.8606885508406725, 0.8142231383664004), 
		("carlow", 0.754603682946357, 0.2952847204328781), 
		("cavan", 0.6745396317053642, 0.6776603968049472), 
		("ennis", 0.1921537229783827, 0.3107446534398351), 
		("cork", 0.3262610088070456, 0.0), 
		("derry", 0.644915932746197, 1.0), 
		("letterkenny", 0.5388310648518815, 0.9806750837413037), 
		("belfast", 1.0, 0.8657562483895903), 
		("enniskillen", 0.5824659727782225, 0.7889719144550373), 
		("galway", 0.18855084067253802, 0.4485957227518681), 
		("tralee", 0.0, 0.12522545735635146), 
		("naas", 0.8350680544435548, 0.4174181911878382), 
		("kilkenny", 0.6749399519615693, 0.24658593146096366), 
		("portlaoise", 0.6749399519615693, 0.35068281370780724), 
		("carrick", 0.45076060848678945, 0.6650347848492657), 
		("limerick", 0.29823859087269816, 0.24658593146096366), 
		("longford", 0.5424339471577262, 0.596753414068539), 
		("dundalk", 0.8987189751801441, 0.6776603968049472), 
		("castlebar", 0.1200960768614892, 0.6302499355836124), 
		("navan", 0.8350680544435548, 0.5671218758052048), 
		("monaghan", 0.7349879903923139, 0.7626900283432105), 
		("tullamore", 0.6365092073658927, 0.4444730739500129), 
		("roscommon", 0.4187349879903923, 0.5735635145581036), 
		("sligo", 0.3506805444355484, 0.7616593661427468), 
		("clonmel", 0.5544435548438751, 0.1571759855707292), 
		("omagh", 0.644515612489992, 0.8660139139397063), 
		("waterford", 0.7269815852682145, 0.11363050760113373), 
		("athlone", 0.5028022417934348, 0.505282143777377), 
		("wexford", 0.8927141713370697, 0.14455037361504766), 
		("bray", 0.9791833466773419, 0.4279824787425921)
]

def get_sims(f="training_data/data.csv"):
    travel_times = []
    
    with open(f, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
        
        i = 0
        for [network, avg_distance, avg_speed] in data:
            assert int(network) == i
            i += 1
            
            time = float(avg_distance) / float(avg_speed)
            travel_times.append(time)
    
    return travel_times
    
    #minimum = min(travel_times)
    #maximum = max(travel_times)
    #for i in range(len(travel_times)):
    #    travel_times[i] = normalize(travel_times[i], minimum / 1.5, maximum)

    #assert min(travel_times) >= 0.00 # and min(travel_times) <  0.01
    #assert max(travel_times) >  0.99 and max(travel_times) <= 1.01    
    #return travel_times

def get_graph_data(f):
	node_list = []
	for (t, x, y) in town_xy_list:
		node = [x, y]
		node_list.append(node)
	
	nodes = torch.tensor(node_list, dtype=torch.float)
	assert nodes.shape == torch.Size([32, 2])
	
	sources = []
	targets = []
	weights = []
	num_edges = 0
	
	with open(f, newline='') as f:
		reader = csv.reader(f)
		data = list(reader)
		data.pop(0)
	
		for [source, target, length, speed, numLanes] in data:
			numLanes = int(numLanes)
			length = float(length)
			num_edges += numLanes
			for i in range(numLanes):
				sources.append(get_town_index(source))
				targets.append(get_town_index(target))
				weights.append([normalize(length, 0.0, 3961.7896208658026)]) # 199.2786993132984
	
	edges = torch.tensor([sources, targets], dtype=torch.long)
	assert edges.shape == torch.Size([2, num_edges])
	
	edge_attr = torch.tensor(weights, dtype=torch.float)
	assert edge_attr.shape == torch.Size([num_edges, 1])
	
	
	graph = Data(x=nodes, edge_index=edges, edge_weight=edge_attr, edge_attr=edge_attr)
	return graph

def get_graph_list(directory="gen_networks/edge_csvs/"):
    graph_list = []
    for i in range(10000):
        f = ''
        if i < 10:
            f = '0'
        if i < 100:
            f = f + '0'
        if i < 1000:
            f = f + '0'
        if i < 10000:
            f = f + '0'
        f = directory + f + str(i) + ".csv"
        
        graph_list.append(get_graph_data(f))
    
    return graph_list

def split_dataset(graph_list, y_list, train_size=0.6, val_size=0.2, test_size=0.2):    
    full_dataset = GraphDataset(graph_list, y_list)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    return train_set, val_set, test_set
        
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
    
    def append(self, graph, y):
        self.graph_list.append(graph)
        self.y_values.append(y)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self, idx):
        return self.__len__()
        
    def get_graphs(self):
        return self.graph_list
    
    def get_y(self):
        return self.y_values

# Define training and evaluation functions
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = F.mse_loss(output, data.y.view(-1, 1))
            total_loss += loss.item()
            all_outputs.append(output.view(-1).cpu())
            all_targets.append(data.y.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(all_targets.numpy(), all_outputs.numpy())
    
    # Calculate R2 Score
    r2 = r2_score(all_targets.numpy(), all_outputs.numpy())
    
    return total_loss / len(loader), mae, r2

def get_model(model, train_dataset, val_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    num_epochs = 100
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    patience_counter = 0
    val_mae = 0
    val_r2 = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader)
        
        if epoch % 25 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')
        
        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter if improvement
        else:
            patience_counter += 1
        
        # Stop early if validation loss doesn't improve for `patience` epochs
        if patience_counter >= patience:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')
            print(f"Early stopping after {epoch} epochs due to no improvement in validation loss.")
            break
    
    return model, val_mae, val_r2

