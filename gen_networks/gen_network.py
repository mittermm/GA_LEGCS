import networkx as nx
import random
import csv
import math
import os

from .csv2xml import csv_to_edg_xml

from .utils import town_list, coordinates, idf_to_str

def gen_strongly_connected_graph(edge_size):

    graph = nx.MultiDiGraph()
    for i in range(len(town_list)):
        graph.add_node(i)
    
    for i in range(edge_size):
        u = random.randint(0,31)
        v = random.randint(0,31)
        while u == v:
            u = random.randint(0,31)
        graph.add_edge(u,v)
    
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
            
    return graph

def graph_to_edge_csv(graph, csv_path):
    edge_list = list(graph.edges)
    i = 0
    while i < len(edge_list) - 1:
        (source1, target1, _) = edge_list[i]
        (source2, target2, _) = edge_list[i+1]
        if source1 == source2 and target1 == target2:
            edge_list.pop(i)
        else:
            i += 1

    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['from', 'to', 'length', 'speed', 'numLanes'])
        for (source, target, lanes) in edge_list:
            (town1, x1, y1) = coordinates[source]
            (town2, x2, y2) = coordinates[target]
            distance = math.dist((x1,y1), (x2,y2))
            num_lanes = lanes + 1
            writer.writerow([town1, town2, str(distance), "10", str(num_lanes)])    

def gen_networks():
    edge_size = 64
    for i in range(10):
        idf = idf_to_str(i)
        graph = gen_strongly_connected_graph(edge_size)
        csv_path = "edge_csvs/" + idf + ".csv"
        graph_to_edge_csv(graph, csv_path)
        xml_path = "edge_xmls/" + idf + ".xml"
        csv_to_edg_xml(csv_path, xml_path)
        net_path = "networks/" + idf + ".net.xml"
        command = "netconvert --node-files=nod.xml --edge-files=" + xml_path + " --output-file=" + net_path
        os.system(command)
    
        edge_size += 1
        if edge_size == 400:
            edge_size = 64

