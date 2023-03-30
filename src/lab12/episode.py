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

import pygame
from pathlib import Path

from lab11.sprite import Sprite
from lab11.turn_combat import CombatPlayer, Combat
from lab11.pygame_ai_player import PyGameAICombatPlayer
from lab11.pygame_human_player import PyGameHumanCombatPlayer


def run_episode(player1: CombatPlayer, player2: CombatPlayer) -> List[Tuple[Tuple[int, int], int, float]]:
    combat = Combat()
    players = [player1, player2]
    observations_actions_rewards = []

    while not combat.gameOver:
        states = list(reversed([(player.health, player.weapon) for player in players]))
        for current_player, state in zip(players, states):
            action = current_player.selectAction(state)
            reward = 0
            if current_player == player1:
                opponent = player2
                if action == 0:
                    reward -= 5
                elif action == 1:
                    opponent.health -= 10
                    if opponent.health <= 0:
                        reward += 100
                        combat.gameOver = True
                    else:
                        reward += 10
                elif action == 2:
                    opponent.health -= 20
                    if opponent.health <= 0:
                        reward += 100
                        combat.gameOver = True
                    else:
                        reward += 20
            else:
                opponent = player1
                if action == 0:
                    reward -= 5
                elif action == 1:
                    opponent.health -= 10
                    if opponent.health <= 0:
                        reward += 100
                        combat.gameOver = True
                    else:
                        reward += 10
                elif action == 2:
                    opponent.health -= 20
                    if opponent.health <= 0:
                        reward += 100
                        combat.gameOver = True
                    else:
                        reward += 20
            observations_actions_rewards.append((state, action, reward))

        combat.newRound()
        combat.takeTurn(player1, player2)

    return observations_actions_rewards