import pygame
import random
import sys 
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import GA




def draw_glow_circle(surface, x, y, radius, color):
    # surface: screen
    # x: circle's x cordinate
    # y: circle's y cordinate
    # radius: circle's radius
    # color: circle's color
    pygame.draw.circle(surface, color, (x, y), radius, width=3)

def create_obstacle(WIDTH, HEIGHT,obstacle_width,obstacle_gap):
    y = random.randint(200, HEIGHT - 200)
    top_rect = pygame.Rect(WIDTH, 0, obstacle_width, y - obstacle_gap // 2)
    bottom_rect = pygame.Rect(WIDTH, y + obstacle_gap // 2, obstacle_width, HEIGHT)
    return top_rect, bottom_rect


def evaluate_ind(population, pop, base_model, dim, gen):
    # population: population size
    # pop: population
    # base_model: neural net architecture 
    # dim: number of parameters 
    # gen: actual generation

    ## initialize Pygame
    pygame.init()

    ## frames
    frames = []

    ## set window frame
    WIDTH, HEIGHT = 600, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Puto el q lo lea")

    # set colors
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    BLACK = (10, 10, 10)
    PINK = (255, 49, 49)

    ## set font
    font = pygame.font.Font("Orbitron-SemiBold.ttf", 30)  # 48 = font size
    #font = pygame.font.SysFont("Arial", 40, bold = True)

    ## set clock
    clock = pygame.time.Clock()
    FPS = 40

    ## circles specs (players)
    player_x = HEIGHT // 4
    player_velocity = np.zeros((population,1))
    player_y = np.ones((population,1))*350
    player_radius = 15 
    gravity = 0.6
    jump_strength = -10

    ## Obstacles specs
    obstacle_width = 50
    obstacle_gap = 100
    obstacle_speed = 5
    obstacles = []


    # Generate the first obstacle
    obstacles.append(create_obstacle(WIDTH, HEIGHT,obstacle_width,obstacle_gap))

    ## set score to zero
    score = np.zeros((population,1))

    ## Create an array of life states
    alive = np.ones((population,), dtype=bool)

    ## neural net output
    Y = np.zeros((population,1))

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update obstacles
        new_obstacles = []

        for top, bottom in obstacles:
            top.x -= obstacle_speed
            bottom.x -= obstacle_speed

            if top.right > 0:
                new_obstacles.append((top, bottom))

            # Detectar colisiones
            for ind in range(population):

                if alive[ind]:
                    if top.collidepoint(player_x, player_y[ind]) or bottom.collidepoint(player_x, player_y[ind]):
                        alive[ind] = False

                    # Sumar puntos cuando pasa un obstáculo
                    if top.right == player_x:

                        score[ind] += 1


        obstacles = new_obstacles

        # Update players
        for ind in range(population):
            if alive[ind]:
                player_velocity[ind] += gravity
                player_y[ind] += player_velocity[ind]

                # Si toca el suelo o el techo
                if player_y[ind] - player_radius > HEIGHT or player_y[ind] + player_radius < 0:
                    alive[ind] = False


        # build neural net inputs
        min_dist = float('inf')
        next_obstacle = None

        for top, bottom in obstacles:
            d = top.x - player_x   # horizontal distance
            if d > 0 and d < min_dist:  # just next obstacles
                min_dist = d
                next_obstacle = (top, bottom)

        if next_obstacle is not None:
            top, bottom = next_obstacle
            dist_x = np.ones((population,1)) * min_dist
            gap_y = np.ones((population,1)) * ((top.height + bottom.y) / 2)
        else:
            dist_x = np.ones((population,1)) * 0
            gap_y = np.ones((population,1)) * (HEIGHT/2)  # valor por defecto

        X = np.hstack([player_y, player_velocity, dist_x, gap_y])

        # neural net action 
        for idx in range(population):
            if alive[idx]:
                model = GA.set_weights_from_vector(base_model, pop[idx,0:dim])
                Y[idx] = model(X[idx, :].reshape(1,4))
                if Y[idx] > 0.5:
                    player_velocity[idx] = jump_strength

        # build a new obstacle
        if not obstacles or obstacles[-1][0].x < WIDTH - 250:
            obstacles.append(create_obstacle(WIDTH, HEIGHT,obstacle_width,obstacle_gap))

        # draw circle and obstacles
        screen.fill(BLACK)
        for ind in range(population):
            if alive[ind]:
                draw_glow_circle(screen, player_x, int(player_y[ind,0]), player_radius, MAGENTA)

        for top, bottom in obstacles:
            pygame.draw.rect(screen, CYAN, top, width=5, border_radius=10)
            pygame.draw.rect(screen, CYAN, bottom, width=5, border_radius=10)

        # Show generation, score and how many individuals are alive
        
        text = font.render(f"Generation: {gen}    Score: {int(max(score))}  Alive: {sum(alive)}", True, PINK)
        screen.blit(text, (30, 30 + 0*40))


        frame = pygame.surfarray.array3d(screen)  # Devuelve un array (ancho, alto, 3)
        frame = np.transpose(frame, (1, 0, 2))   # Lo rota para que quede (alto, ancho, canales)
        frames.append(frame)

        ## break
        if sum(alive) == 0:
        	pygame.quit()
        	break

        pygame.display.flip()
        clock.tick(FPS)

    ## return score
    return score, frames 
