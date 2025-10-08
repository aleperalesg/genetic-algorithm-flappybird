# Evolving a Neural Network with a Genetic Algorithm to Play Flappy Bird. 
A population of 100 neural networks is evolved over generations, with each network representing a potential player. The networks use a sigmoid activation function in the output layer.
Each network receives four inputs extracted from the game environment: 

-Vertical coordinate of the player
-Vertical velocity of the player
-Horizontal distance to the nearest obstacle
-Vertical position of the obstacle gap

The MLP outputs a value between 0 and 1. If the value exceeds 0.5, the player jumps. The GA optimizes the networks based on the score achieved by each individual.
Operators used:

-Crossover: 70%
-Mutation: 30%
-Replication: 0%

The selection mechanism is tournament selection with a tournament size of 3. 
The following GIF shows the **optimization process of the MLP population** over generations using a genetic algorithm.  
Each individual represents a neural network trying to improve its performance in the Flappy Bird environment:
![Demo](output.gif)

  
