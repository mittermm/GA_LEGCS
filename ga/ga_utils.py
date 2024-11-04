import random
import math
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import csv

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

town_coordinates = [
        ("dublin",3004,-2551),
        ("antrim",2968,-879),
        ("craigavon",2832,-1208),
        ("carlow",2567,-3222),
        ("cavan",2367,-1738),
        ("ennis",1162,-3162),
        ("cork",1497,-4368),
        ("derry",2293,-487),
        ("letterkenny",2028,-562),
        ("belfast",3180,-1008),
        ("enniskillen",2137,-1306),
        ("galway",1153,-2627),
        ("tralee",682,-3882),
        ("naas",2768,-2748),
        ("kilkenny",2368,-3411),
        ("portlaoise",2368,-3007),
        ("carrick",1808,-1787),
        ("limerick",1427,-3411),
        ("longford",2037,-2052),
        ("dundalk",2927,-1738),
        ("castlebar",982,-1922),
        ("navan",2768,-2167),
        ("monaghan",2518,-1408),
        ("tullamore",2272,-2643),
        ("roscommon",1728,-2142),
        ("sligo",1558,-1412),
        ("clonmel",2067,-3758),
        ("omagh",2292,-1007),
        ("waterford",2498,-3927),
        ("athlone",1938,-2407),
        ("wexford",2912,-3807),
        ("bray",3128,-2707)
]

def get_towns():
    towns = []
    for (t,_,_) in town_coordinates:
        towns.append(t)
    return towns

def normalize(n, MIN_SIM, MAX_SIM):
    return ((n - MIN_SIM) / (MAX_SIM - MIN_SIM))

def get_town_index(town):
    i = 0
    for (t, _, _) in town_xy_list:
        if t == town:
            return i
        i += 1
    return -1
    
    

def get_graph_data(edge_list):
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

    for [source, target, length, speed, numLanes] in edge_list:
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

def get_random_edge():
    u = random.randint(0,31)
    v = random.randint(0,31)
    while u == v:
        u = random.randint(0,31)
    
    (source, s_x, s_y) = town_coordinates[u]
    (target, t_x, t_y) = town_coordinates[v]    
    
    distance = math.dist([s_x, s_y], [t_x, t_y])
    
    return [source, target, distance, 10, 1]

def get_coordinates(town):
    for (t, x, y) in town_coordinates:
        if t == town:
            return [x,y]

def heal(edge_list):
    graph = nx.MultiDiGraph()
    
    for (t, _, _) in town_coordinates:
        graph.add_node(t)
    
    for (s,t,_,_,_) in edge_list:
        graph.add_edge(s,t)
    
    if nx.is_strongly_connected(graph):
        return edge_list
    
    while not nx.is_strongly_connected(graph):
        components = list(nx.strongly_connected_components(graph))
    
        i = random.randint(0, len(components)-1)   
        comp1 = list(components[i])
    
        j = i
        while i == j:
            j = random.randint(0, len(components)-1)   
        
        comp2 = list(components[j])
    
        try:
            nx.shortest_path(graph, comp1[0], comp2[0])
    
            i = random.randint(0, len(comp2)-1)
            u2 = comp2[i]
            i = random.randint(0, len(comp1)-1)
            v2 = comp1[i]
            graph.add_edge(u2, v2)
    
        except:
            i = random.randint(0, len(comp1)-1)
            u1 = comp1[i]
            i = random.randint(0, len(comp2)-1)
            v1 = comp2[i]
            graph.add_edge(u1,v1)
    
    edge_list = list(graph.edges)
    i = 0
    while i < len(edge_list) - 1:
        (source1, target1, _) = edge_list[i]
        (source2, target2, _) = edge_list[i+1]
        if source1 == source2 and target1 == target2:
            edge_list.pop(i)
        else:
            i += 1
    
    new_edge_list = []
    for (source, target, lanes) in edge_list:
        source_coord = get_coordinates(source)
        target_coord = get_coordinates(target)
        distance = math.dist(source_coord, target_coord)
        num_lanes = lanes + 1
        new_edge_list.append([source, target, str(distance), "10", str(num_lanes)])
    
    return new_edge_list
    
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

def get_initial_population(interval=range(10000), directory="gen_networks/edge_csvs/"):
    population = []
    for i in interval:
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
        
        with open(f, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data.pop(0)
            
            population.append(data)
    
    return population
