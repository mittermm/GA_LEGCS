from gnn.gnn_utils import get_model, normalize_list, normalize, GraphDataset
from gnn.gnn import GNN
from ga.ga_utils import get_initial_population, get_sims, get_graph_data
from ga.ga import GeneticAlgorithm
import os
import csv
import xml.etree.ElementTree as ET
from config.run_simulation import run_simulation
from torch.utils.data import random_split
import numpy as np
import random

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
                        "from": f,
                        "to": t,
                        "length": l,
                        "speed": s,
                        "numLanes": n
                    })
        i += 1
    tree = ET.ElementTree(tree)
    tree.write(edge_path)
    
    command = "netconvert --node-files=" + node_path + " --edge-files=" + edge_path + " --output-file=" + net_path
    os.system(command)
    
    avg_distance, avg_speed = run_simulation(config_path)
    print("network " + simID
        + "\n\taverage distance: " + str(avg_distance)
        + "\n\taverage speed:    " + str(avg_speed))
    
    time = avg_distance * avg_speed
    return time    
    

class GA_LEGCS(GeneticAlgorithm):
    
    def __init__(self, initial_population, sims_list, generations, mutation_rate):
        self.population = initial_population
        self.pop_size = len(self.population)
        self.graph_list = []
        for edge_list in self.population:
            self.graph_list.append(get_graph_data(edge_list))
        self.sims_list = sims_list
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness = sims_list.copy()
    
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
        best_sim = min(self.fitness)
        best_gen = -1
        best_genome = self.population[self.fitness.index(best_sim)]
        
        print("Initialisation")
        print("best sim ", best_sim, " from Gen", best_gen)
        
        
        for generation in range(self.generations):
            print("Generation ", generation)
                       
            # train GNN model
            normalized_sims, n_min, n_max = normalize_list(self.sims_list)
            train_set, val_set = random_split(GraphDataset(self.graph_list, normalized_sims), [0.8, 0.2])
            r2 = 0
            model = get_model(GNN(), train_set, val_set)
            
            # evolve
            self.evolve()
            for i in range(self.pop_size):
                model.eval()
                self.fitness[i] = model(get_graph_data(self.population[i])).item()
            
            # simulate
            top_predictions = list(np.argsort(self.fitness)[:int(0.01 * self.pop_size)])
            for i in top_predictions:
                simID = "gen" + str(generation) + "genome" + str(i)
                sim = simulate(self.population[i], simID)
                print("pred ", self.fitness[i], " to sim ", normalize(sim, n_min, n_max))
                
                if sim < best_sim:
                    best_sim = sim
                    best_gen = generation
                    best_genome = self.population[i]
                
                self.graph_list.append(get_graph_data(self.population[i]))
                self.sims_list.append(sim)
                self.fitness[i] = normalize(sim, n_min, n_max)

            print("best sim ", best_sim, " from Gen", best_gen)
        
        return best_genom, best_sim

initial_population = get_initial_population(range(1000))
sims_list = get_sims()
sims_list = sims_list[:1000]
ga = GA_LEGCS(initial_population, sims_list, 100, 0.3)
best_genome, best_fitness = ga.run()

print(best_genome)
print(best_fitness)
