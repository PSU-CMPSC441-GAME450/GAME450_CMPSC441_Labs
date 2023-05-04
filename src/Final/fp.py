##########################################################################
#####COPIED LAB 11 AGENT_ENVIRONMENT CONTENT TO HERE AND ADDED TO IT######
##########################################################################
import sys
import pygame
import random
import numpy as np
from pathlib import Path
from bresenham import bresenham
sys.path.append(str((Path(__file__) / ".." / "..").resolve().absolute()))
from lab2.cities_n_routes import get_randomly_spread_cities, get_routes
from lab11.pygame_ai_player import PyGameAIPlayer
from lab11.sprite import Sprite
from lab11.pygame_combat import run_pygame_combat
from lab11.pygame_human_player import PyGameHumanPlayer
from lab11.landscape import get_landscape, get_combat_bg
from lab7.ga_cities import make_cities, get_elevation
from lab2.cities_n_routes import get_randomly_spread_cities, get_routes
from transformers import AutoTokenizer, AutoModelForCausalLM

pygame.font.init()
game_font = pygame.font.SysFont("Comic Sans MS", 15)

####LANGUAGE TRANSFORMATION FOR TEXT GENERATION####
import torch
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def get_landscape_surface(size):
    landscape = get_landscape(size)
    print("Created a landscape of size", landscape.shape)
    pygame_surface = pygame.surfarray.make_surface(landscape[:, :, :3])
    return pygame_surface, landscape


def get_combat_surface(size):
    landscape = get_combat_bg(size)
    print("Created a landscape of size", landscape.shape)
    pygame_surface = pygame.surfarray.make_surface(landscape[:, :, :3])
    return pygame_surface


def setup_window(width, height, caption):
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption(caption)
    return window


def displayCityNames(city_locations, city_names):
    for i, name in enumerate(city_names):
        text_surface = game_font.render(str(i) + " " + name, True, (0, 0, 150))
        screen.blit(text_surface, city_locations[i])

#Check validity of routes
def is_valid_route(routes, current, new, cityList):
    start_city = cityList[current]
    end_city = cityList[new]
    for route in routes:
        if np.array_equal(route[:2], [start_city, end_city]) or \
           np.array_equal(route[:2], [end_city, start_city]):
            return True
    return False

def find_route(routes, current, next):
    start_city = cities[current]
    end_city = cities[next]
    for i, route in enumerate(routes):
        if [start_city, end_city] in route or [end_city, start_city] in route:
            return i
    return 0


def get_cost(routes, game_map, routenum):
    route_coordinate = routes[routenum]
    path = list(bresenham(route_coordinate[0][0],route_coordinate[0][1],route_coordinate[1][0],route_coordinate[1][1])) 
    cost = game_map[tuple(zip(*path))].sum() % 200
    return cost

class State:
    def __init__(
        self,
        current_city,
        destination_city,
        travelling,
        encounter_event,
        cities,
        routes,
        budget,
    ):
        self.current_city = current_city
        self.destination_city = destination_city
        self.travelling = travelling
        self.encounter_event = encounter_event
        self.cities = cities
        self.routes = routes
        self.budget = 700

if __name__ == "__main__":
    size = width, height = 640, 480
    black = 1, 1, 1
    start_city = 0
    end_city = 9
    sprite_path = "assets/lego.png"
    sprite_speed = 1

    screen = setup_window(width, height, "Game World Gen Practice")

    landscape_surface, landscape = get_landscape_surface(size)
    combat_surface = get_combat_surface(size)
    city_names = [
        "Morkomasto",
        "Morathrad",
        "Eregailin",
        "Corathrad",
        "Eregarta",
        "Numensari",
        "Rhunkadi",
        "Londathrad",
        "Baernlad",
        "Forthyr",
    ]

    ## cities = result of GA
    cities = make_cities(size, len(city_names))
    #cities = get_randomly_spread_cities(size, len(city_names))
    routes = get_routes(cities)

    random.shuffle(routes)
    routes = routes[:10]

    player_sprite = Sprite(sprite_path, cities[start_city])

    #player = PyGameHumanPlayer()

    """ Add a line below that will reset the player variable to 
    a new object of PyGameAIPlayer class."""

    player = PyGameAIPlayer()

    state = State(
        current_city=start_city,
        destination_city=start_city,
        travelling=False,
        encounter_event=False,
        cities=cities,
        routes=routes,
        budget = 700,
    )

    while True:
        action = player.selectAction(state)
        if 0 <= int(chr(action)) <= 9:
            if int(chr(action)) != state.current_city and not state.travelling and is_valid_route(routes, state.current_city, int(chr(action)), cities):
                route = find_route(routes, state.current_city, int(chr(action)))
                print('You have arrived to the city.')
                prompt = "I have finally arrived to the next elven city. The journey was long but now I can rest and reflect on my fantastical journey thus far."
                input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
                output = model.generate(input_ids, max_length=250, pad_token_id=tokenizer.eos_token_id, do_sample=True)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print("Generated text:", generated_text)                
                state.budget -= get_cost(routes, landscape, route)
                if(state.budget <= 0):
                    print("Your budget is empty. Good try.")
                    break
                print("BUDGET: ", state.budget)

                start = cities[state.current_city]
                state.destination_city = int(chr(action))
                destination = cities[state.destination_city]
                player_sprite.set_location(cities[state.current_city])
                state.travelling = True
                print(
                    "Travelling from", state.current_city, "to", state.destination_city
                )

        screen.fill(black)
        screen.blit(landscape_surface, (0, 0))

        for city in cities:
            pygame.draw.circle(screen, (255, 0, 0), city, 5)

        for line in routes:
            pygame.draw.line(screen, (255, 0, 0), *line)

        displayCityNames(cities, city_names)
        if state.travelling:
            state.travelling = player_sprite.move_sprite(destination, sprite_speed)
            state.encounter_event = random.randint(0, 1000) < 2
            if not state.travelling:
                print('Arrived at', state.destination_city)

        if not state.travelling:
            encounter_event = False
            state.current_city = state.destination_city

        if state.encounter_event:
            temp = run_pygame_combat(combat_surface, screen, player_sprite)
            state.encounter_event = False
            #if loss combat, end
            if (temp == 0 or temp == -1):
                break
        else:
            player_sprite.draw_sprite(screen)
        pygame.display.update()
        if state.current_city == end_city:
            print('You have reached the end of the game!')
            break