import numpy
import GA
from fitness import func
num_weights = int(input('Enter the required input values'))
equation_inputs=num_weights*[4]
sol_per_pop = 8
num_parents_mating = 63
pop_size = (sol_per_pop,num_weights) 
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)
best_outputs = []
num_generations = 100
def genetic(num_weights,func):
    for generation in range(num_generations):

        print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        fitness = func(equation_inputs)
        print("Fitness")
        print(fitness)

        best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        # The best result in the current iteration.
        print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    
        # Selecting the best parents in the population for mating.
        parents = GA.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)
        print("Parents")
        print(parents)

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        print("Crossover")
        print(offspring_crossover)

        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)
        print("Mutation")
        print(offspring_mutation)
        

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.


# Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))
    print("Best solution : ", new_population[best_match_idx, :])
    print("Best solution fitness : ", fitness[best_match_idx])
#------------------------------------------------------------------------------#
genetic(num_weights,func)
