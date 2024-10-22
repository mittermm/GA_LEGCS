from gnn.gnn import GNN
import torch
from ga.ga import GeneticAlgorithm2

#gnn_file = 'gnn/trained_gnn_model.pth'
gnn_file = 'gnn/model-02.pth'

model = GNN()
model.load_state_dict(torch.load(gnn_file))


ga = GeneticAlgorithm2(1000, 0.1, 50, model)
best_genome, best_fitness = ga.run()

print(best_genome)
print(best_fitness)
