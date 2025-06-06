from gnn.gnn_utils import get_model, normalize_list, normalize, GraphDataset
from gnn.gnn import GNN
from ga.ga_utils import get_initial_population, get_sims, get_graph_data, heal, get_towns
from ga.ga import GeneticAlgorithm
import os
import csv
import xml.etree.ElementTree as ET
from config.run_simulation import run_simulation
from torch.utils.data import random_split
import numpy as np
import random
from torch.utils.data import ConcatDataset
import logging

log_file = "output/ga_lecgs.logfile"
logging.basicConfig(filename=log_file,
                    filemode='a',
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.DEBUG)

def simulate(edge_list, simID):
    node_path = "config/workspace/nod.xml"
    edge_path = "config/workspace/edg.xml"
    net_path = "config/workspace/current_network.net.xml"
    config_path = "config/workspace/base.sumo.cfg"

    tree = ET.Element("edges")
    i = 0
    
    for [f, t, l, s, n] in edge_list:
        edge_id = f"edge_{i}"
        ET.SubElement(tree,
                    "edge",
                    id=edge_id,
                    attrib={
                        "from": str(f),
                        "to": str(t),
                        "length": str(l),
                        "speed": str(s),
                        "numLanes": str(n)
                    })
        i += 1
    tree = ET.ElementTree(tree)
    tree.write(edge_path)
    
    command = "netconvert --node-files=" + node_path + " --edge-files=" + edge_path + " --output-file=" + net_path
    os.system(command)
    
    avg_distance, avg_speed = run_simulation(config_path)
    print("network " + simID + " with " + str(len(edge_list)) + " streets:"
        + "\n\taverage distance: " + str(avg_distance)
        + "\n\taverage speed:    " + str(avg_speed))
    
    time = avg_distance / avg_speed
    return time    
    
def determine_fitness(travel_time, edge_list, max_street_length):
    # return travel_time + max(0, len(edge_list) - 100) * 0.001
    total_length = 0
    for [_, _, l, _, _] in edge_list:
        total_length += float(l)
    normalized_length = total_length / max_street_length
    
    return travel_time + normalized_length

class GA_LEGCS(GeneticAlgorithm):
    
    def __init__(self, initial_population, sims_list, generations, mutation_rate):
        self.population = initial_population
        self.pop_size = len(self.population)
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.minmax = [min(sims_list) / 1.5, max(sims_list)]
        print("normalisation: min = ", self.minmax[0], " max = ", self.minmax[1])
        self.max_street_length = self.determine_street_length_normalizer()
        self.initial_graph_dataset = GraphDataset([],[])
        self.runtime_graph_dataset = GraphDataset([],[])
        self.fitness = []
        for i in range(self.pop_size):
            y = normalize(sims_list[i], self.minmax[0], self.minmax[1])
            self.initial_graph_dataset.append(get_graph_data(self.population[i]), y)
            self.fitness.append(determine_fitness(y, self.population[i], self.max_street_length))
    
    def determine_street_length_normalizer(self):
        max_length = 0
        for edge_list in self.population:
            current_length = 0
            for [_,_,l,_,_] in edge_list:
                current_length += float(l)
            max_length = max(max_length, current_length)
        return max_length
    
    def update_fitness(self, travel_times):
        assert len(self.population) == self.pop_size
        assert len(travel_times) == self.pop_size
        self.fitness = [-1] * self.pop_size
        
        for i in range(self.pop_size):
            self.fitness[i] = determine_fitness(travel_times[i], self.population[i], self.max_street_length)
        
        assert min(self.fitness) > -1
    
    def edge_based_crossover(self, edges_1, edges_2):
        
        # Randomly choose a subset of edges
        crossover_point = len(edges_1) // 2
        selected_edges_2 = edges_2[:crossover_point]
        
        child1 = edges_1[:crossover_point] + edges_2[crossover_point:]
        child2 = edges_2[:crossover_point] + edges_1[crossover_point:]
        
        child1 = heal(child1)
        child2 = heal(child2)
        
        return child1, child2
    
    def sub_graph_crossover(self, edges_1, edges_2):
        shuffled_towns = sorted(get_towns(), key=lambda x: random.random())
        crossover_point = len(shuffled_towns) // 2
        
        nodes_1 = shuffled_towns[:crossover_point]
        nodes_2 = shuffled_towns[crossover_point:]
        
        child1 = []
        child2 = []
        
        remaining_edges = []
        for edge in edges_1:
            if edge[0] in nodes_1 and edge[1] in nodes_1:
                child1.append(edge)
            elif edge[0] in nodes_2 and edge[1] in nodes_2:
                child2.append(edge)
            else:
                remaining_edges.append(edge)
        
        for edge in edges_2:
            if edge[0] in nodes_1 and edge[1] in nodes_1:
                child2.append(edge)
            elif edge[0] in nodes_2 and edge[1] in nodes_2:
                child1.append(edge)
            else:
                remaining_edges.append(edge)
        
        b = True
        for edge in remaining_edges:
            if b:
                child1.append(edge)
                b = False
            else:
                child2.append(edge)
                b = True

        child1 = heal(child1)
        child2 = heal(child2)
        
        return child1, child2
    
    def evolve(self):
        new_population = []
        
        # elitism
        elitism = 0.01 * self.pop_size
        elite_indices = list(np.argsort(self.fitness)[:int(elitism)])
        for i in elite_indices:
            new_population.append(self.population[i])
        
        while len(new_population) < self.pop_size:
            
            # selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # crossover (implemented in ga.ga)
            child1, child2 = self.crossover(parent1, parent2)
            #child1, child2 = self.edge_based_crossover(parent1, parent2)
            
            # mutation (implemented in ga.ga)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            new_population.append(child2)
        
        new_population = new_population[:self.pop_size]
        self.population = new_population
    
    def tournament_selection(self, k=3):
        selected = random.sample(range(self.pop_size), k)
        selected_fitness = [self.fitness[index] for index in selected]
        
        for i in selected:
            if self.fitness[i] == min(selected_fitness):
                return self.population[i]
    
    def run(self):
        i = self.fitness.index(min(self.fitness))
        best_gen = -1
        best_sim = self.initial_graph_dataset.get(i).y.item()
        best_genome = self.population[i]
        best_fitness = self.fitness[i]
        
        print("Initialisation")
        print("best sim ", best_sim, " with ", len(best_genome), " streets from Gen", best_gen)
        print("best fitness ", best_fitness) 
        logging.info("Initialisation: best fitness = " + str(best_fitness) + " with " + str(len(best_genome)) + " streets")
        
        for generation in range(self.generations):
            print("Generation ", generation)
            logging.info("starting generation " + str(generation))
            self.mutation_rate = self.mutation_rate - 0.004
            
            # train GNN model every 10 generations
            if generation % 10 == 0:
                i_train_set, i_val_set = random_split(self.initial_graph_dataset, [0.8, 0.2])
                r_train_set, r_val_set = random_split(self.runtime_graph_dataset, [0.8, 0.2])
                
                train_set = ConcatDataset([i_train_set, r_train_set])
                val_set = ConcatDataset([i_val_set, r_val_set])
                model, val_mae, val_r2 = get_model(GNN(), train_set, val_set)
                
                logging.info("new GNN model trained, mae = " + str(val_mae) + ", r2 = " + str(val_r2))
            
            # evolve
            self.evolve()
            predictions = []
            for i in range(self.pop_size):
                model.eval()
                predictions.append(model(get_graph_data(self.population[i])).item())
            self.update_fitness(predictions)
            
            # simulate
            top_predictions = list(np.argsort(self.fitness)[:int(0.1 * self.pop_size)])
            #top_predictions = list(np.argsort(self.fitness)[:int(0.03 * self.pop_size)])
            #random_preds = []
            #while len(random_preds) < 5:
            #    idx = random.randint(0, self.pop_size - 1)
            #    if idx not in top_predictions:
            #        random_preds.append(idx)
            sim_indices = top_predictions # + random_preds
            for i in sim_indices:
                simID = "gen" + str(generation) + "genome" + str(i)
                current_edge_list = self.population[i].copy()
                sim = simulate(self.population[i], simID)
                y = normalize(sim, self.minmax[0], self.minmax[1])
                print(sim, " normalised with min = ", self.minmax[0], " max = ", self.minmax[1], " to ", y)
                current_fitness = determine_fitness(y, self.population[i], self.max_street_length)
                print("pred fitness ", self.fitness[i], " to sim fitness ", current_fitness)
                logging.info("simulated " + simID + ": fitness = " + str(current_fitness) + " (predicted " + str(self.fitness[i]) + ") with " + str(len(self.population[i])) + " streets")
                self.fitness[i] = current_fitness
                self.runtime_graph_dataset.append(get_graph_data(self.population[i]), y) 

                
                if current_fitness < best_fitness:
                    print("We found a new best network!")
                    best_sim = sim
                    best_gen = generation
                    best_genome = self.population[i]
                    best_fitness = current_fitness
                    
                    with open("output/" + simID + ".csv", 'w') as file:
                        writer = csv.writer(file)
                        writer.writerows(self.population[i])                

            print("best sim ", best_sim, " with ", len(best_genome), " streets from Gen", best_gen)
            print("best fitness: ", best_fitness)
            logging.info("best fitness " + str(best_fitness) + " from generation " + str(best_gen))
        
        return best_genome, best_sim

# population_size = 500
# initial_population = get_initial_population(range(population_size))
# sims_list = get_sims()
# sims_list = sims_list[:population_size]
# ga = GA_LEGCS(initial_population, sims_list, 100, 0.5)
# best_genome, best_fitness = ga.run()
# 
# print(best_genome)
# print(best_fitness)
