from gnn.gnn_utils import get_model, normalize_list, normalize, GraphDataset
from gnn.gnn import GNN
from ga.ga_utils import get_initial_population, get_sims, get_graph_data, heal, get_towns
from ga_lecgs import GA_LEGCS, simulate, determine_fitness
import os
import csv
import xml.etree.ElementTree as ET
from config.run_simulation import run_simulation
from torch.utils.data import random_split
import numpy as np
import random
from torch.utils.data import ConcatDataset
import logging

log_file = "output/ga_nolecgs.logfile"
logging.basicConfig(filename=log_file,
                    filemode='a',
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.DEBUG)
   
class GA_NOLEGCS(GA_LEGCS):
    
    def __init__(self, initial_population, sims_list, generations, mutation_rate):
        self.population = initial_population
        self.pop_size = len(self.population)
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.minmax = [min(sims_list) / 1.5, max(sims_list)]
        print("normalisation: min = ", self.minmax[0], " max = ", self.minmax[1])
        self.max_street_length = self.determine_street_length_normalizer()        
        self.fitness = []
        for i in range(self.pop_size):
            y = normalize(sims_list[i], self.minmax[0], self.minmax[1])
            self.fitness.append(determine_fitness(y, self.population[i]))
            
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
            self.mutation_rate = self.mutation_rate - 0.04
            
            # evolve
            self.evolve()
            
            # simulate
            for i in range(self.pop_size):
                simID = "gen" + str(generation) + "genome" + str(i)
                sim = simulate(self.population[i], simID)
                y = normalize(sim, self.minmax[0], self.minmax[1])
                current_fitness = determine_fitness(y, self.population[i])
                print("sim fitness ", current_fitness)
                logging.info("simulated " + simID + ": fitness = " + str(current_fitness) " with " + str(len(self.population[i])) + " streets")
                self.fitness[i] = current_fitness
                
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

# population_size = 100
# initial_population = get_initial_population(range(population_size))
# sims_list = get_sims()
# sims_list = sims_list[:population_size]
# ga = GA_NOLEGCS(initial_population, sims_list, 10, 0.1)
# best_genome, best_fitness = ga.run()
# 
# print(best_genome)
# print(best_fitness)
