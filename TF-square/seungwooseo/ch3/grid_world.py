from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from itertools import product


class GridWorld:
    def __init__(self, height: int, width: int,
                 pos_a: List, pos_a_: List,
                 pos_b: List, pos_b_: List,
                 init_pos: List):
        self.height = height
        self.width = width
        self.pos_a = np.array(pos_a, dtype=np.int)
        self.pos_a_ = np.array(pos_a_, dtype=np.int)
        self.pos_b = np.array(pos_b, dtype=np.int)
        self.pos_b_ = np.array(pos_b_, dtype=np.int)
        self.actions = {'up': np.array([-1, 0], dtype=np.int), 'down': np.array([1, 0], dtype=np.int),
                        'left': np.array([0, -1], dtype=np.int), 'right': np.array([0, 1], dtype=np.int)}

        self.pos = np.array(init_pos, dtype=np.int)

    def __call__(self, action: str, pos=None) -> Tuple[np.array, float]:
        if pos is not None:
            self.pos = pos
        r = 0
        move = self.actions[action]
        if (self.pos == self.pos_a).all():
            r = 10.
            self.pos = self.pos_a_
        elif (self.pos == self.pos_b).all():
            r = 5.
            self.pos = self.pos_b_
        else:
            self.pos += move
            if self.pos[0] == self.height or self.pos[0] < 0:
                r = -1.
                self.pos -= move
            elif self.pos[1] == self.width or self.pos[1] < 0:
                r = -1.
                self.pos -= move

        return self.pos, r


class RandomAgent:
    def __init__(self, height, width, init_pos, gamma):
        self.cum_r = 0.
        self.value = np.zeros((height, width))
        self.pos = init_pos
        self.gamma = gamma

    def __call__(self, world):
        action = ['up', 'down', 'left', 'right'][np.random.choice(4)]
        new_pos, r = world(action)
        self.value[self.pos[0], self.pos[1]] += 0.25 * (r + self.gamma * self.value[new_pos[0], new_pos[1]])
        self.pos = new_pos

        return self.value


def loop(max_iter=10):
    world = GridWorld(5, 5, [0, 1], [4, 1], [0, 3], [2, 3], [0, 0])
    rand_agent = RandomAgent(5, 5, [0, 0], 0.9)
    value = rand_agent.value
    for _ in tqdm(range(max_iter)):
        for _ in tqdm(range(10)):
            rand_agent(world)
    print(value)


def mdp_rand_pi(height: int, width: int,
                pos_a: List, pos_a_: List,
                pos_b: List, pos_b_: List,
                gamma: float):

    value = np.zeros((height, width))
    world = GridWorld(height, width, pos_a, pos_a_, pos_b, pos_b_, [0, 0])
    for _ in tqdm(range(100000)):
        new_value = np.zeros_like(value)
        for i, j in product(range(height), range(width)):
            for act in ['up', 'down', 'left', 'right']:
                new_pos, r = world(act, pos=np.array([i, j]))
                new_value[i, j] += 0.25 * (r + gamma * value[new_pos[0], new_pos[1]])
        if np.allclose(value, new_value):
            break
        value = new_value.copy()
    return value


if __name__ == '__main__':
    value = mdp_rand_pi(5, 5, [0, 1], [4, 1], [0, 3], [2, 3], 0.9)
    print(value)
    # loop()
