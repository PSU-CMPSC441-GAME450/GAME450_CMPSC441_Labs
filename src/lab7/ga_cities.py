"""
Lab 7: Realistic Cities 

In this lab you will try to generate realistic cities using a genetic algorithm.
Your cities should not be under water, and should have a realistic distribution across the landscape.
Your cities may also not be on top of mountains or on top of each other.
Create the fitness function for your genetic algorithm, so that it fulfills these criterion
and then use it to generate a population of cities.

Please comment your code in the fitness function to explain how are you making sure each criterion is 
fulfilled. Clearly explain in comments which line of code and variables are used to fulfill each criterion.
"""
import matplotlib.pyplot as plt
import pygad
import numpy as np

import sys
from pathlib import Path

sys.path.append(str((Path(__file__) / ".." / ".." / "..").resolve().absolute()))

from src.lab5.landscape import elevation_to_rgba, get_elevation


def game_fitness(cities, idx, elevation, size):
    fitness = 0.0001  # Do not return a fitness of 0, it will mess up the algorithm.
    """
    Create your fitness function here to fulfill the following criteria:
    1. The cities should not be under water
    2. The cities should have a realistic distribution across the landscape
    3. The cities may also not be on top of mountains or on top of each other
    """
    #COMMENT THIS FUNCTION
    coordinates = solution_to_cities(cities, size)      # list of coordinates for each city
    comparison = coordinates                            # duplicate list to compare
    allowablySpaced = 1                                 # variable to check if all of the cities are spaced enough from each other
    allSpaced = 1                                       # variable to check if all of the cities are spaced from each other
    allSpacedWide = 1                                   # variable to check if all of the cities are spaced widely from each other
    magnitude = 5                                       # magnitude of separation variables 

    track1 = 0                                          # track outer loop
    for c in coordinates:                               # iterate through each city
        track2 = 0                                          # track inner loop

        spaced = 1                                      # variable to check if one city is spaced from all others
        if (elevation[c[0]][c[1]] < 0.55):              # increase fitness for optimal elevation
            if(elevation[c[0]][c[1]] > 0.45):
                fitness += 2        
        elif (elevation[c[0]][c[1]] < 0.60):            # increase fitness for acceptable elevation
            if(elevation[c[0]][c[1]] > 0.40):
                fitness += 1.5        
        elif (elevation[c[0]][c[1]] < 0.7):             # increase fitness for essential elevation
            if(elevation[c[0]][c[1]] > 0.35):
                fitness += 1
        if (elevation[c[0]][c[1]] > 0.7 or elevation[c[0]][c[1]] < 0.35):       # adjust to unacceptable fit for unusable elevation
                fitness -= 10000
        for comp in comparison:                         # compare each city to its neighbors
                if (track1 != track2):                  # do not compare a city to itself
                    if(abs(c[0] - comp[0]) > 90 & abs(c[1]-comp[1]) > 80):    # increase fitness for great diffusion
                        fitness += 3
                    elif(abs(c[0] - comp[0]) > 70 & abs(c[1]-comp[1]) > 60):  # increase fitness for acceptale diffusion
                        fitness += 2.75
                    elif(abs(c[0] - comp[0]) > 50 & abs(c[1]-comp[1]) > 40):  # increase fitness for acceptale diffusion
                        fitness += 2.55
                        allSpacedWide = 0                                       # all cities cannot be spaced widely
                    elif(abs(c[0] - comp[0]) > 30 & abs(c[1]-comp[1]) > 20):  # increase fitness for acceptale diffusion
                        fitness += 2            
                        allSpacedWide = 0                                       # all cities cannot be spaced widely
                        spaced = 0                                              # cities are not spaced
                    elif(abs(c[0] - comp[0]) < 30 & abs(c[1]-comp[1]) < 20):  # decrease fitness for poor diffusion
                        fitness -= 3
                        spaced = 0 
                        allSpaced = 0                                             # cities are not spaced
                        allSpacedWide = 0                                       # all cities cannot be spaced widely
                    if(abs(c[0] - comp[0]) < 25 & abs(c[1]-comp[1]) < 15):      # decrease fitness for horrible diffusion
                        fitness -= 5
                        spaced = 0                                              # cities are not spaced
                        allSpacedWide = 0                                       # cities cannot be spaced widely
                        allSpaced = 0                                           # cities cannot be spaced in general
                        allowablySpaced = 0
                    if (spaced == 0):                                           # decrease fitness if not spaced
                        fitness -= 20
                    else:                                                       # increase fitness if spaced
                        fitness += 10                       
                    if(abs(c[0] - comp[0]) > 90 & abs(c[1]-comp[1]) > 80):                  # set value for extra large spacing
                        magnitude = min(magnitude, 5)
                        if(abs(c[0] - comp[0]) < 90 & abs(c[1]-comp[1]) < 80):              # set value for no extra large spacing 
                            magnitude = min(magnitude, 4)
                            if(abs(c[0] - comp[0]) < 70 & abs(c[1]-comp[1]) < 60):          # set value for no large spacing
                                magnitude = min(magnitude, 3)
                                if(abs(c[0] - comp[0]) < 50 & abs(c[1]-comp[1]) < 40):      # set value for no medium spacing
                                    magnitude = min(magnitude, 2)
                                    if(abs(c[0] - comp[0]) < 30 & abs(c[1]-comp[1]) < 20):  # set value for no small spacing
                                        magnitude = 1
                track2 += 1                     
        track1 += 1
    if (allowablySpaced > 0):       # increase fitness if all cities are spaced from each other well enough to meet requirement
        fitness += 10
    else:
        fitness -= 10000
    if (magnitude == 1):            # decrease fitness if citites are too close
        fitness -= 100
    elif (magnitude == 2):          # decrease fitness if citites are minimally spread
        fitness -= 50
    elif (magnitude == 3):          # increase fitness if citites are spread decently
        fitness += 100
    elif (magnitude == 4):          # increase fitness largely if citites are spread well
        fitness += 250
    elif (magnitude == 5):          # increase greatly if citites are spread far
        fitness += 500  
    if (allSpaced > 0):             # increase fitness tremedously if all cities are spaced from each other
        fitness += 75
    if (allSpacedWide > 0):         # further inflate fitness if all cities are spaced widely (best case scenario)
        fitness += 1000
    #if(track1 == 10 & track2 == 10):
        #print(magnitude)
    return fitness


def setup_GA(fitness_fn, n_cities, size):
    """
    It sets up the genetic algorithm with the given fitness function,
    number of cities, and size of the map

    :param fitness_fn: The fitness function to be used
    :param n_cities: The number of cities in the problem
    :param size: The size of the grid
    :return: The fitness function and the GA instance.
    """
    num_generations = 100
    num_parents_mating = 10

    solutions_per_population = 300
    num_genes = n_cities

    init_range_low = 0
    init_range_high = size[0] * size[1]

    parent_selection_type = "sss"
    keep_parents = 10

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_fn,
        sol_per_pop=solutions_per_population,
        num_genes=num_genes,
        gene_type=int,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
    )

    return fitness_fn, ga_instance


def solution_to_cities(solution, size):
    """
    It takes a GA solution and size of the map, and returns the city coordinates
    in the solution.

    :param solution: a solution to GA
    :param size: the size of the grid/map
    :return: The cities are being returned as a list of lists.
    """
    cities = np.array(
        list(map(lambda x: [int(x / size[0]), int(x % size[1])], solution))
    )
    return cities


def show_cities(cities, landscape_pic, cmap="gist_earth"):
    """
    It takes a list of cities and a landscape picture, and plots the cities on top of the landscape

    :param cities: a list of (x, y) tuples
    :param landscape_pic: a 2D array of the landscape
    :param cmap: the color map to use for the landscape picture, defaults to gist_earth (optional)
    """
    cities = np.array(cities)
    plt.imshow(landscape_pic, cmap=cmap)
    plt.plot(cities[:, 1], cities[:, 0], "r.")
    plt.show()


if __name__ == "__main__":
    print("Initial Population")

    size = 100, 100
    n_cities = 10
    elevation = []
    """ initialize elevation here from your previous code"""
    elevation = get_elevation(size)
    # normalize landscape
    elevation = np.array(elevation)
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    landscape_pic = elevation_to_rgba(elevation)

    # setup fitness function and GA
    fitness = lambda cities, idx: game_fitness(
        cities, idx, elevation=elevation, size=size
    )
    fitness_function, ga_instance = setup_GA(fitness, n_cities, size)

    # Show one of the initial solutions.
    cities = ga_instance.initial_population[0]
    cities = solution_to_cities(cities, size)
    show_cities(cities, landscape_pic)

    # Run the GA to optimize the parameters of the function.
    ga_instance.run()
    ga_instance.plot_fitness()
    print("Final Population")

    # Show the best solution after the GA finishes running.
    cities = ga_instance.best_solution()[0]
    cities_t = solution_to_cities(cities, size)
    plt.imshow(landscape_pic, cmap="gist_earth")
    plt.plot(cities_t[:, 1], cities_t[:, 0], "r.")
    plt.show()
    print(fitness_function(cities, 0))
