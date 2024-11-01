from gnn.gnn_utils import get_model, normalize_list, normalize, GraphDataset
from gnn.gnn import GNN
from ga.ga_utils import get_initial_population, get_sims, get_graph_data, heal, get_towns
from ga_legcs import GA_LEGCS, simulate, determine_fitness
import os
import csv
import xml.etree.ElementTree as ET
from config.run_simulation import run_simulation
from torch.utils.data import random_split
import numpy as np
import random
from torch.utils.data import ConcatDataset
import logging

log_file = "output/ga_nosims.logfile"
logging.basicConfig(filename=log_file,
                    filemode='a',
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.DEBUG)

class GA_NOSIMS(GA_LEGCS):
      
    def run(self):
        print("no sims run")
        i = self.fitness.index(min(self.fitness))
        best_gen = -1
        best_sim = self.initial_graph_dataset.get(i).y.item()
        best_genome = self.population[i]
        best_fitness = self.fitness[i]
        
        print("Initialisation")
        print("best sim ", best_sim, " with ", len(best_genome), " streets from Gen", best_gen)
        print("best fitness ", best_fitness) 
        logging.info("Initialisation: best fitness = " + str(best_fitness) + " with " + str(len(best_genome)) + " streets (sim = " + str(best_sim) + ")")
        
        train_set, val_set = random_split(self.initial_graph_dataset, [0.8, 0.2])
        model, val_mae, val_r2 = get_model(GNN(), train_set, val_set)
        logging.info("GNN model trained, mae = " + str(val_mae) + ", r2 = " + str(val_r2))
        
        for generation in range(self.generations):
            print("Generation ", generation)
            logging.info("starting generation " + str(generation))
            self.mutation_rate = self.mutation_rate - 0.004
            
            # evolve
            self.evolve()
            predictions = []
            for i in range(self.pop_size):
                model.eval()
                predictions.append(model(get_graph_data(self.population[i])).item())
            self.update_fitness(predictions)
            
            i = self.fitness.index(min(self.fitness))
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_genome = self.population[i]
                best_gen = generation
            
            print("best fitness: ", best_fitness, " with ", len(best_genome), " from Gen ", best_gen)
            logging.info("best fitness " + str(best_fitness) + " from generation " + str(best_gen))
        
        logging.info("GA done, start simulations")
        top_predictions = list(np.argsort(self.fitness)[:10])
        best_fitness = float('inf')
        for i in top_predictions:
            simID = "genome" + str(i)
            sim = simulate(self.population[i], simID)
            y = normalize(sim, self.minmax[0], self.minmax[1])
            current_fitness = determine_fitness(y, self.population[i])
            print("pred fitness ", self.fitness[i], " to sim fitness ", current_fitness)
            logging.info("simulated " + simID + ": fitness = " + str(current_fitness) + " (predicted " + str(self.fitness[i]) + ") with " + str(len(self.population[i])) + " streets")
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_sim = sim
                best_genome = self.population[i]
                
                with open("output/" + simID + ".csv", 'w') as file:
                    writer = csv.writer(file)
                    writer.writerows(self.population[i])

        print("best fitness found = " + str(best_fitness) + " with " + str(len(best_genome)) + " streets (sim = " + str(best_sim) + ")")
        logging.info("best fitness found = " + str(best_fitness) + " with " + str(len(best_genome)) + " streets (sim = " + str(best_sim) + ")")
        
        return best_genome, best_sim

# population_size = 500
# initial_population = get_initial_population(range(population_size))
# sims_list = get_sims()
# sims_list = sims_list[:population_size]
# ga = GA_NOSIMS(initial_population, sims_list, 10000, 0.5)
# best_genome, best_fitness = ga.run2()
# 
# print(best_genome)
# print(best_fitness)
