import random
import itertools
from collections import defaultdict, deque
from turn_combat import CombatPlayer

class PyGameAIPlayer:
    def __init__(self):
        self.path = []
        self.path_index = 0

    def selectAction(self, state):
        if state.travelling:
            return ord(" ")
        elif state.current_city == state.destination_city:
            valid_cities = []
            for r in state.routes:
                if r[0] == state.current_city:
                    valid_cities.append(r[1])
                elif r[1] == state.current_city:
                    valid_cities.append(r[0])
            if valid_cities:
                destination = random.choice(valid_cities)
                self.path = self.find_path(state.current_city, destination, state.routes)
                self.path_index = 0
                return ord(str(destination))
            else:
                return ord(" ")
        elif self.path_index < len(self.path):
            next_city = self.path[self.path_index]
            self.path_index += 1
            return ord(str(next_city))
        else:
            state.destination_city = state.current_city
            return ord(" ")

    def find_path(self, start, end, routes):
        graph = defaultdict(list)
        for r in routes:
            graph[r[0]].append(r[1])
            graph[r[1]].append(r[0])

        q = deque()
        visited = [False] * len(graph)
        path = [-1] * len(graph)

        q.append(start)
        visited[start] = True

        while q:
            u = q.popleft()
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    path[v] = u
                    q.append(v)

        p = []
        if path[end] != -1:
            while end != -1:
                p.insert(0, end)
                end = path[end]
        return p





        #return random.randint(48, 57) # ASCII values for digits 0-9

class PyGameAICombatPlayer(CombatPlayer):
    def __init__(self, name):
        super().__init__(name)

    def weapon_selecting_strategy(self):
        weapon = random.randint(0, 2)
        self.weapon = weapon
        return self.weapon











