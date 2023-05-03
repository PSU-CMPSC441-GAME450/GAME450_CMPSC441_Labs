""" Create PyGameAIPlayer class here"""
import pygame
import random
from lab11.turn_combat import CombatPlayer


class PyGameAIPlayer:
        def __init__(self) -> None:
            pass

        def selectAction(self, state):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                else:
                    random_number = random.randint(0,9)
                    action = ord(str(random_number))
                    return action
            return ord(str(state.current_city))  # Not a safe operation for >10 cities

""" Create PyGameAICombatPlayer class here"""
class PyGameAICombatPlayer(CombatPlayer):
    def __init__(self, name):
        super().__init__(name)

    def weapon_selecting_strategy(self):
            self.weapon = random.randint(0,2)
            return self.weapon            
