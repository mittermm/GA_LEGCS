from .ga_utils import get_graph_data, get_random_edge, heal
import random
import numpy as np

class GeneticAlgorithm:

    def __init__(self, pop_size, mutation_rate, generations, gnn_model):
        self.pop_size = pop_size  # Population size
        self.mutation_rate = mutation_rate  # Mutation rate
        self.generations = generations  # Number of generations
        self.gnn_model = gnn_model
        self.population = self.init_population()
        
    def init_population(self):
        """Initialize the population with random genomes."""
        population = []
        
        # max edges = 400
        # min edges = 64
        
        j = -1
        if self.pop_size < 400-64:
            j = int((400 - 64) / self.pop_size)
        
        for i in range(self.pop_size):
            edge_counter = -1
            
            if j == -1:
                edge_counter = 64 + i % 400
            else:
                edge_counter = 64 + i * j
             
            edge_list = []
            for k in range(edge_counter):
                edge_list.append(get_random_edge())
            
            edge_list = heal(edge_list)
            population.append(edge_list)
        
        return population

    def tournament_selection(self, k=3):
        """Select a genome using tournament selection."""
        selected = random.sample(self.population, k)
        selected_fitness = [self.fitness(genome) for genome in selected]
        min_fitness = float('inf')
        i = -1
        for j in range(k):
            if min_fitness > selected_fitness[j]:
                min_fitness = selected_fitness[j]
                i = j
        return selected[i] # Select the best genome (minimization)

    def crossover(self, parent1, parent2):
        interval1 = random.randint(0,9)
        interval2 = random.randint(interval1+1, 10)
        
        length1 = len(parent1)
        length2 = len(parent2)
        
        child1 = []
        child2 = []
        
        limit11 = int((interval1 / 10) * length1)
        limit12 = int((interval1 / 10) * length2)
        limit21 = int((interval2 / 10) * length1)
        limit22 = int((interval2 / 10) * length2)
        
        #             l11  l21
        # parent1 [####|####|####] 
        #
        #                l12     l22
        # parent2 [#######|#######|#######]
        #
        # child1  [####|#######|####]
        # child2  [#######|####|#######]
        
        i = 0
        while i < limit11:
            child1.append(parent1[i])
            i += 1
        
        i = limit12
        while i < limit22:
            child1.append(parent2[i])
            i += 1
        
        i = limit21
        while i < length1:
            child1.append(parent1[i])
            i += 1
        
        i = 0
        while i < limit12:
            child2.append(parent2[i])
            i += 1
        
        i = limit11
        while i < limit21:
            child2.append(parent1[i])
            i += 1
        
        i = limit22
        while i < length2:
            child2.append(parent2[i])
            i += 1
        
        child1 = heal(child1)
        child2 = heal(child2)
        
        return child1, child2

    def mutate(self, edge_list):
    
        # add edge
        if random.random() < self.mutation_rate:
            for j in range(5):
                [source, target, distance, speed, lanes] = get_random_edge()
                added = False
                for i in range(len(edge_list)):
                    [s,t,d,sp,l] = edge_list[i]
                    if s == source and t == target:
                        edge_list[i] = [s,t,d,sp, str(int(l)+1) ]
                        added = True
                
                if not added:
                    edge_list.append([source, target, distance, speed, lanes])
        
        # remove edge
        if random.random() < self.mutation_rate:
            for i in range(5):
                edge_to_remove = random.choice(range(len(edge_list)))
                edge_list.pop(edge_to_remove)
    
        # remove_edges = []
        # add_edges = []
        # i = 0
        # while i < len(edge_list):
        #     if random.random() < self.mutation_rate:
        #         edge_list.pop(i)
        #         r = random.random()
        #         if r < 0.33:
        #             add_edges.append(get_random_edge())
        #         elif r < 0.67:
        #             add_edges.append(get_random_edge())
        #             add_edges.append(get_random_edge())
        #     else:
        #         i += 1
        
        edge_list = heal(edge_list)
        return edge_list

    def evolve(self):
        """Evolve the population using selection, crossover, and mutation."""
        new_population = []
        for _ in range(self.pop_size // 2):
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.extend([child1, child2])

        # Replace the old population with the new one
        self.population = new_population

    def run(self, patience=10):
        """Run the genetic algorithm for a given number of generations with early stopping."""
        best_genome = None
        best_fitness = float('inf')
        patience_counter = 0  # Keeps track of how long since the last fitness improvement
    
        for generation in range(self.generations):
            # Evolve the population
            self.evolve()
    
            # Track whether any improvement was made this generation
            improvement = False
    
            # Evaluate the population
            for genome in self.population:
                current_fitness = self.fitness(genome)
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_genome = genome
                    improvement = True  # We found a better genome this generation
    
            # Check for early stopping
            if improvement:
                patience_counter = 0  # Reset patience if we made an improvement
            else:
                patience_counter += 1  # Increment patience if no improvement
    
            print(f"Generation {generation}, Best Fitness: {best_fitness}")
    
            # Stop if we've hit the patience limit without improvement
            if patience_counter >= patience:
                print(f"Stopping early at generation {generation} due to no improvement for {patience} generations.")
                break

        return best_genome, best_fitness


    def fitness(self, edge_list):
        self.gnn_model.eval()
        avg_travel_time = self.gnn_model(get_graph_data(edge_list)).item()
        
        num_lanes = 0
        for (_, _, _, _, lanes) in edge_list:
            num_lanes += int(lanes)
        
        num_lanes = (1.0 * num_lanes) / 400.0
        
        return avg_travel_time # + 0.6 * num_lanes
    
    

class GeneticAlgorithm2(GeneticAlgorithm):
    
    def roulette_wheel_selection(self, fitness_values, num_selections=2):
        
        max_f = max(fitness_values) + 1e-6
        inversed_fitness = []
        for f in fitness_values:
            inversed_fitness.append(max_f - f)
        
        total = sum(inversed_fitness)
        probabilities = []
        for f in inversed_fitness:
            probabilities.append(f / total)
        
        selected_indices = np.random.choice(self.pop_size, num_selections, probabilities)
        
        return [self.population[i] for i in selected_indices]
    
    def evolve(self):
        """Evolve the population using selection, crossover, and mutation."""
        new_population = []
        
        fitness_values = []
        for genome in self.population:
            fitness_values.append(self.fitness(genome))
        
        for _ in range(self.pop_size // 2):
            # Selection
            [parent1, parent2] = self.roulette_wheel_selection(fitness_values, 2)
    
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
    
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
    
            new_population.extend([child1, child2])
    
        # Replace the old population with the new one
        self.population = new_population

class GeneticAlgorithm3(GeneticAlgorithm):
    def __init__(self, pop_size, mutation_rate, initial_population, initial_fitness):
        self.pop_size = pop_size  # Population size
        self.mutation_rate = mutation_rate  # Mutation rate
        self.population = initial_population
        self.fitness = initial_fitness
        
    def tournament_selection(self, k=3):
        """Select a genome using tournament selection."""
        selected = random.sample(range(self.pop_size), k)
        selected_fitness = [self.fitness[index] for index in selected]
        
        for i in selected:
            if self.fitness[i] == min(selected_fitness):
                return self.population[i]    
        