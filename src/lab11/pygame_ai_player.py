import random

class PyGameAIPlayer:
    def __init__(self) -> None:
        pass

    def selectAction(self, state):
        return random.randint(48, 57) # ASCII values for digits 0-9

class PyGameAICombatPlayer:
    def __init__(self, name):
        super().__init__(name)

    def weapon_selecting_strategy(self):
        choices = ["s", "a", "f"]
        return random.choice(choices)


