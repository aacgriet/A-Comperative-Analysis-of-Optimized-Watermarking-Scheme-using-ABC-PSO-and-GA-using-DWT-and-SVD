import numpy
import GA
from fitness import func

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
num_weights = 64
equation_inputs =[1.8340971974432712, 8.147593046774087, 6.630235927501817, 5.85189068411585, 4.8044735686722175, 6.557465080275829, 6.120694410775569, 4.205331125196269, 5.226881011697675, 4.733100061398467, 6.262259858202323, 4.4127687312866275, 4.58161247305241, 2.5966430230557167, 2.306688202895164, 4.839103762898533, 5.630787055082777, 3.5861977728881587, 6.1934092795976685, 6.384698010251916, 5.549498086837806, 4.529247271022309, 4.492724402301866, 6.942583533147453, 7.743833581977869, 2.6161282814425295, 2.8775942430229784, 6.560527618416758, 5.058220688852217, 4.239784877890976, 6.199022662844103, 4.4880887006111605, 3.7783219615131793, 3.4178505671733075, 6.837846532192707, 6.527535459402626, 4.456983995105128, 4.793175434171657, 6.548010539197725, 6.170996789320986, 4.146537356886773, 4.676952128156739, 5.910471447711865, 4.769520612137483, 3.7886454267369634, 7.299790116363878, 5.017960292756532, 6.275514493190572, 4.231666353802281, 6.146352013189814, 3.114061451632947, 4.394373326941629, 3.804135290975952, 4.646915071676957, 5.929742380016509, 5.557596390201964, 5.68934245251085, 2.754001384733983, 7.346934264071505, 5.64340395734305, 6.283067476412483, 5.710956860772021, 4.667335429319024, 5.056865194483114]

equation_inputs = func(equation_inputs )
# Number of the weights we are looking to optimize.


"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10
num_parents_mating = 5

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.uniform(low=1.0, high=10.0, size=pop_size)
print('new',new_population)
num_generations = 1000
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(equation_inputs, new_population)

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
