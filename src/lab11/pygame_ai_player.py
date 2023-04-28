import random
import itertools
from turn_combat import CombatPlayer

class PyGameAIPlayer:
    def selectAction(self, state):
        return random.randint(48, 57) # ASCII values for digits 0-9


class PyGameAICombatPlayer(CombatPlayer):
    def __init__(self, name):
        super().__init__(name)

    def weapon_selecting_strategy(self):
        weapon = random.randint(0, 2)
        self.weapon = weapon
        return self.weapon











