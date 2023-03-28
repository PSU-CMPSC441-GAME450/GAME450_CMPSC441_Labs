''' 
Lab 12: Beginnings of Reinforcement Learning
We will modularize the code in pygrame_combat.py from lab 11 together.

Then it's your turn!
Create a function called run_episode that takes in two players
and runs a single episode of combat between them. 
As per RL conventions, the function should return a list of tuples
of the form (observation/state, action, reward) for each turn in the episode.
Note that observation/state is a tuple of the form (player1_health, player2_health).
Action is simply the weapon selected by the player.
Reward is the reward for the player for that turn.
'''
from pathlib import Path
import sys
sys.path.append(str((Path(__file__) / ".." / "..").resolve().absolute()))
from lab11.pygame_combat import run_turn, PyGameComputerCombatPlayer
from lab11.turn_combat import Combat
from lab11.pygame_ai_player import PyGameAICombatPlayer

def run_episode(player, opponent):
    currentGame = Combat()
    data = []

    while not currentGame.gameOver:
        reward = run_turn(currentGame, player, opponent)
        data.append(((player.health, opponent.health), player.weapon, reward))

    return data
    
if __name__ == "__main__":
    player = PyGameAICombatPlayer("AI")
    opponent = PyGameComputerCombatPlayer("Computer")

    print(run_episode(player, opponent))

