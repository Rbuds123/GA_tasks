import numpy as np

def initialize_population(pop_size, chromosome_length):
    return np.random.choice([0, 1], size=(pop_size, chromosome_length))
    
def fitness(chromosome):
    # Separate the chromosome into four 8-bit blocks
    chromosome_1 = chromosome[:8]
    chromosome_2 = chromosome[8:24]
    chromosome_3 = chromosome[24:32]
    
    # Maximize 1s in the blue blocks
    Part_1 = np.sum(chromosome_1) + np.sum(chromosome_3)
    
    # Minimize 1s in the green block
    part_2 = np.sum(chromosome_2)
    
    # Define the fitness score: more weight to blue maximization, less for green minimization
    return Part_1 - part_2

def roulette_wheel_selection(population, fitnesses):
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness  # Shift all fitnesses to be non-negative

    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        probabilities = np.ones(len(population)) / len(population)  # Equal probability if all fitnesses are zero
    else:
        probabilities = fitnesses / total_fitness

    return population[np.random.choice(len(population), p=probabilities)]


def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return offspring1, offspring2

def double_point_crossover(parent_1, parent_2):
    point_1, point_2 = sorted(np.random.choice(range(1, len(parent_1)), size=2, replace=False))
    
    offspring1 = np.concatenate([parent_1[:point_1], parent_2[point_1:point_2], parent_1[point_2:]])
    offspring2 = np.concatenate([parent_2[:point_1], parent_1[point_1:point_2], parent_2[point_2:]])
    
    return offspring1, offspring2

def mutate(chromosome, mutation_rate):
    mutation_mask = np.random.rand(len(chromosome)) < mutation_rate
    chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
    return chromosome

def new_generation(population, mutation_rate):
    new_pop = []
    fitnesses = [fitness(chromo) for chromo in population]
    for _ in range(len(population) // 2):  # Half the population will be parents
        parent1 = roulette_wheel_selection(population, fitnesses)
        parent2 = roulette_wheel_selection(population, fitnesses)
        offspring1, offspring2 = single_point_crossover(parent1, parent2)
        new_pop.extend([mutate(offspring1, mutation_rate), mutate(offspring2, mutation_rate)])
    return new_pop

def genetic_algorithm(pop_size, chromosome_length, mutation_rate, generations):
    population = initialize_population(pop_size, chromosome_length)

    # Variables to store best fitness and the generation it was found
    best_solution = None
    best_fitness = float('-inf')
    best_generation = -1

    for generation in range(generations):
        population = new_generation(population, mutation_rate)
        current_best_solution = max(population, key=fitness)
        current_best_fitness = fitness(current_best_solution)

        # If we find a better solution, update best solution and generation
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            best_generation = generation

        print(f"Generation {generation}, Best solution: {current_best_solution}, Fitness: {current_best_fitness}")

    print(f"Best solution found at generation {best_generation} with fitness {best_fitness}")
    return best_solution, best_generation

best_solution, best_generation = genetic_algorithm(pop_size=100, chromosome_length=32, mutation_rate=0.1, generations=5000)
print(f"Best solution found: {best_solution} at generation {best_generation} with a fitness of {fitness(best_solution)}")