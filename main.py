import numpy as np 
import GA
import evaluate
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

## parameters
crossover_p = 0.70
mutation_p = 0.30
replication_p = 0.00
tournament_s = 3
lower_bound = -1
upper_bound = 1
generations = 500
population = 100            
gen = 0


## create population
base_model,pop, dim = GA.generate_population(population, lower_bound, upper_bound)
frames = []
best_ind_fit = 0 
while gen < generations:


    ## evaluate population
    fitness, frames_ = evaluate.evaluate_ind(population, pop, base_model,dim,gen)

    ## get frames for a video
    frames = frames + frames_

    print(f"Generación {gen}, mejor fitness: {fitness.max():.4f}")

    ## get best individual for elitism
    best_ind, best_ind_fit_eli = GA.elitism(pop, fitness, dim)

    ## get operators crossover, mutation and replication 
    opts = GA.get_operators(population,crossover_p, mutation_p, replication_p)

    ## new population
    new_pop = np.zeros_like(pop)

    ## create next generation
    count = 0
    for opt in opts:    
        if opt == "m":

            ## apply mutation 
            new_pop[count] = GA.mutation(pop, fitness, tournament_s, lower_bound, upper_bound)
            count +=1

        elif opt == "r":

            ## apply replication
            new_pop[count] = GA.replication(pop, fitness, tournament_s)
            count +=1

            ## apply crossover
        elif opt == "c":
            f_offspring, s_offspring = GA.crossover(pop, fitness, tournament_s, dim)
            new_pop[count] = f_offspring
            new_pop[count+1] = s_offspring
            count +=2

    ## update new population
    pop = new_pop

    ## apply elitism
    if best_ind_fit_eli > best_ind_fit:
        pop[0,0:dim] = best_ind
        best_ind_fit = best_ind_fit_eli
    gen += 1    

    pil_frames = [Image.fromarray(frame.astype("uint8"), "RGB") for frame in frames]

    ## Save gif
    pil_frames[0].save(
        "output.gif",
        save_all=True,
        append_images=pil_frames[1:],  
        duration=40,  
        loop=0        
    )



