import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def generate_population(population, lower_bound, upper_bound):
    #population: population size
    #lower_bound: int 
    #upper_bound: int

    ## set architecture
    base_model = create_mlp()

    ## get number of parameters
    weights  = base_model.get_weights()
    dim = sum(w.size for w in weights)

    ## create population
    pop = np.random.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(population, dim)
    )

    return base_model, pop, dim 


def create_mlp():

    ## set model
    model = models.Sequential()
    model.add(layers.Input(shape=(4,)))      # input layer
    #model.add(layers.Dense(5, activation='relu'))  # hidden layer
    model.add(layers.Dense(1, activation='sigmoid')) # output layer
    model.compile(optimizer="adam", loss="mse")  

    return model
     
def set_weights_from_vector(model, vector):
    # model: mlp model
    # vector: flattened vector of weigghts
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(vector[idx:idx+size].reshape(shape))
        idx += size
    
    model.set_weights(new_weights)
    return model



def tournament_selection(fitness, tournament_s):
    # fitness: array with fitness of population  
    # tournament_s: tournament size

    idx = np.arange(len(fitness))
    # randomly select ‘tournament_s’ individuals without repetition
    tournament = np.random.choice(idx, size=tournament_s, replace=False)
    
    # return best induvidual's index
    winner = tournament[fitness[tournament].argmax()]  
    return winner

def get_operators(population,crossover_p, mutation_p, replication_p):
    #population: Population size
    #crossover_p: Crossover probability
    #mutation_p: mutation probability
    #replication_p: replication probability

    ## operators list 
    operators = []
    count = 0

    ## operators names
    elements = ["c","m","r"] 

    ## get operators
    while count < population:
        if count == population-1:
            opt = np.random.choice(["m","r"], p=[0.75, 0.25])
        else:

            opt = np.random.choice(elements, p=[crossover_p,mutation_p, replication_p])
        
        operators.append(opt)

        if opt == "c":
            count += 2
        elif opt == "m":
            count += 1
        elif opt == "r":
            count += 1

    return operators


def mutation(pop,fitness,tournament_s,lower_bound, upper_bound):
    #pop: population
    #fitness: fitness array:
    #tournament_s: tournament size
    #lower_bound: int 
    #upper_bound: int

    ## select individual 
    individual = pop[tournament_selection(fitness, tournament_s)]

    ## get mutation 
    mutation = np.random.normal(0, 0.1*(upper_bound-lower_bound), size=individual.shape)

    ## add mutation
    ind_mutated = individual + mutation

    ## return new individual
    return np.clip(ind_mutated, lower_bound, upper_bound)

def replication(pop, fitness, tournament_s):
    #pop: population
    #fitness: fitness array:
    #tournament_s: tournament size

    ## return a selected individual
    return pop[tournament_selection(fitness, tournament_s)]

def crossover(pop,fitness,tournament_s,dim):

    ## Select the first parent
    f_parent_idx = tournament_selection(fitness, tournament_s)

    ## loop to select a second diferent parent 
    loop = True
    while True:
        s_parent_idx = tournament_selection(fitness, tournament_s)
        if f_parent_idx != s_parent_idx:
            break

    ## crossover point 
    point = np.random.randint(1, dim)

    ## get parents
    f_parent, s_parent = pop[f_parent_idx], pop[s_parent_idx]

    ## crossover
    f_offspring = np.concatenate((f_parent[:point],s_parent[point:]))
    s_offspring = np.concatenate((s_parent[:point],f_parent[point:]))

    ## return offsprings
    return f_offspring, s_offspring

def elitism(pop, fitness, dim):
    best_ind_idx = np.argmax(fitness)
    best_ind_fit = np.max(fitness)
    return pop[best_ind_idx,0:dim], best_ind_fit
