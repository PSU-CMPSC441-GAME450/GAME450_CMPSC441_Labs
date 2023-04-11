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


def run_episode(player1, player2):
    players = [player1, player2]
    player1.health, player2.health = 100, 100  # reset player health
    observations = []
    
    while True:
        player1_health, player2_health = player1.health, player2.health
        states = [(player1_health, player1.weapon), (player2_health, player2.weapon)]
        actions = [player.selectAction(state) for player, state in zip(players, states)]
        rewards = [0, 0]

        if actions[0] == actions[1]:
            # both players selected same weapon
            rewards = [0, 0]
        elif (actions[0] + 1) % 3 == actions[1]:
            # player 2 wins
            player2.health -= 20
            rewards = [-20, 20]
        else:
            # player 1 wins
            player1.health -= 20
            rewards = [20, -20]

        observation = (player1_health, player2_health)
        observations.append((observation, actions[0], rewards[0]))
        observations.append(((player2_health, player1_health), actions[1], rewards[1]))

        if player1.health <= 0 or player2.health <= 0:
            # game over
            break
        
    return observations

# need to collect all of the actions
# return state, action, reward 

