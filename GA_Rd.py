import random as rd

def initialize_population(pop_size, chromosome_length):
    return [[rd.choice([0, 1]) for _ in range(chromosome_length)] for _ in range(pop_size)]

def fitness(chromosome):
    target = [1] * 8 + [0] * 16 + [1] * 8
    
    # Calculate fitness by summing matches with the target pattern
    fitness_score = sum(1 for i in range(len(chromosome)) if chromosome[i] == target[i])
    
    return fitness_score



def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return rd.choices(population, weights=probabilities, k=1)[0]


def single_point_crossover(parent1, parent2):
    crossover_point = rd.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]

#def double_point_crossover(parent_1, parent_2):
    assert len(parent_1) == len(parent_2), "Parents must be of the same length"
    
    point_1 = rd.randint(1, len(parent_1) - 2)
    point_2 = rd.randint(point_1 + 1, len(parent_1) - 1)

    offspring1 = parent_1[:point_1] + parent_2[point_1:point_2] + parent_1[point_2:]
    offspring2 = parent_2[:point_1] + parent_1[point_1:point_2] + parent_2[point_2:]

    return offspring1, offspring2

def mutate(chromosome, mutation_rate):
    return [gene if rd.random() > mutation_rate else 1 - gene for gene in chromosome]


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

best_solution, best_generation = genetic_algorithm(pop_size=10, chromosome_length=32, mutation_rate=0.1, generations=500)
print(f"Best solution found: {best_solution} at generation {best_generation}")