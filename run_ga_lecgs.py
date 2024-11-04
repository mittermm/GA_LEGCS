from ga_legcs import GA_LEGCS
from ga.ga_utils import get_initial_population, get_sims

population_size = 500
initial_population = get_initial_population(range(population_size))
sims_list = get_sims()
sims_list = sims_list[:population_size]
ga = GA_LEGCS(initial_population, sims_list, 100, 0.5)
best_genome, best_fitness = ga.run()

print(best_genome)
print(best_fitness)
