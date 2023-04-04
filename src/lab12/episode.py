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
sys.path.append('/path/to/lab11')

from lab11.turn_combat import CombatPlayer, Combat


def run_episode(player1, player2):
    currentGame = Combat()
    player1.reset()
    player2.reset()

    observation = (player1.health, player2.health)
    episode = [(observation, None, 0)]

    while not currentGame.gameOver:
        action1 = player1.selectAction(observation)
        action2 = player2.selectAction(observation[::-1])[::-1]

        currentGame.newRound()
        currentGame.takeTurn(player1, player2)
        reward1 = currentGame.checkWin(player1, player2)
        reward2 = -reward1

        observation = (player1.health, player2.health)
        episode.append((observation, action1, reward1))
        episode.append((observation[::-1], action2, reward2))

    return episode
    pass
    # return state, action, history 
