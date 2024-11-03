import random as rd
import numpy as np

# Initialize population
def initialize_population(size, chromosome_length=32):
    population = []
    for _ in range(size):
        population.append(rd.choices([0, 1], k=chromosome_length))
    return population

# Define the target chromosome pattern
target_chromosome = [1, 1, 1, 1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 
                     1, 1, 1, 1, 1, 1, 1, 1]

# Updated fitness function
def fitness_function(individual):
    # Count how many bits match the target chromosome
    fitness = sum(1 for i, bit in enumerate(individual) if bit == target_chromosome[i])
    return fitness

# Roulette wheel selection
def roulette_wheel_selection(pop, fitness_values):
    total_fitness = sum(fitness_values)
    normalized = [f / total_fitness for f in fitness_values]  # Normalize fitness
    cumulative_fitness = np.cumsum(normalized)
    
    selected_parents = []
    for _ in range(len(pop)):
        r = rd.uniform(0, 1)
        for i, individual in enumerate(pop):
            if r <= cumulative_fitness[i]:
                selected_parents.append(individual)
                break
    return selected_parents

# Single-point crossover
def single_point_crossover(parent_a, parent_b):
    cut_point = rd.randint(1, len(parent_a) - 1)
    offspring_1 = parent_a[:cut_point] + parent_b[cut_point:]
    offspring_2 = parent_b[:cut_point] + parent_a[cut_point:]
    return offspring_1, offspring_2

# Mutation function
def mutate(individual, mutation_rate=0.3):
    for idx in range(len(individual)):
        if rd.random() < mutation_rate:
            individual[idx] = 1 - individual[idx]  # Flip bit
    return individual

# Create a new generation
def new_generation(population, mutation_rate=0.3):
    fitness_values = [fitness_function(ind) for ind in population]
    parents = roulette_wheel_selection(population, fitness_values)
    
    # Create new offspring through crossover and mutation
    new_population = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            offspring_1, offspring_2 = single_point_crossover(parents[i], parents[i+1])
            new_population.append(mutate(offspring_1, mutation_rate))
            new_population.append(mutate(offspring_2, mutation_rate))
    return new_population

# Main Genetic Algorithm
def genetic_algorithm(pop_size=10, chromosome_length=32, mutation_rate=0.3, generations=100):
    population = initialize_population(pop_size, chromosome_length)
    
    # Track the best solution
    best_solution = None
    best_fitness = float('-inf')
    best_generation = -1

    for generation in range(generations):
        population = new_generation(population, mutation_rate)
        current_best_solution = max(population, key=fitness_function)
        current_best_fitness = fitness_function(current_best_solution)
        
        # Update best solution if current one is better
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            best_generation = generation
            
        print(f"Generation {generation}, Best solution: {current_best_solution}, Fitness: {current_best_fitness}")
        
    print(f"Best solution found at generation {best_generation} with fitness {best_fitness}")
    return best_solution, best_generation

# Run the genetic algorithm
best_solution, best_generation = genetic_algorithm(pop_size=10, chromosome_length=32, mutation_rate=0.1, generations=6000)
print(f"Best solution found: {best_solution} at generation {best_generation} with a fitness of {fitness_function(best_solution)}")
