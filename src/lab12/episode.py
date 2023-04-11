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
import sys
import random
sys.path.append('/path/to/lab11')

from lab11.turn_combat import CombatPlayer, Combat


def run_episode(player1, player2):
 

    player1_health = 100
    player2_health = 100


    history = []

<<<<<<< HEAD
# need to collect all of the actions
# return state, action, reward 

=======

    while player1_health > 0 and player2_health > 0:
        p1_weapon = player1.select_weapon()
        p2_weapon = player2.select_weapon()

        p1_damage = random.randint(5, 20)
        p2_damage = random.randint(5, 20)

        player1_health -= p2_damage
        player2_health -= p1_damage

        p1_reward = p2_damage if player1_health <= 0 else p2_damage - p1_damage
        p2_reward = p1_damage if player2_health <= 0 else p1_damage - p2_damage

        history.append(((player1_health, player2_health), (p1_weapon, p2_weapon), (p1_reward, p2_reward)))

    return history
    # return state, action, history 
>>>>>>> c0526fda864094b7cbed65b8d01e26b339cbadc8
