import itertools
import random
import sys
import math
import pygame
import numpy as np
import pygad
import matplotlib.pyplot as plt

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid 
from pathfinding.finder.a_star import AStarFinder

from perlin_noise import PerlinNoise


from pathlib import Path
sys.path.append(str((Path(__file__)/'..'/'..').resolve().absolute()))
from src.lab2.cities_n_routes import get_randomly_spread_cities, get_routes

from src.lab4.rock_paper_scissor import Player

from src.lab5.landscape import elevation_to_rgba

from src.lab11.sprite import Sprite
from src.lab11.pygame_combat import run_pygame_combat
from src.lab11.pygame_human_player import PyGameHumanPlayer
from src.lab11.landscape import get_landscape, get_combat_bg
from src.lab11.pygame_ai_player import PyGameAIPlayer
from src.lab11.turn_combat import CombatPlayer

