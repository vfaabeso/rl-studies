import numpy as np
import random
from maze import Maze
from tqdm import tqdm

# the main algorithm
def tabular_dyna_q(
    env: Maze,
    max_steps: int = 10000, n: int = 10,
    epsilon: float = 0.2, alpha: float = 0.1, gamma: float = 0.95,
    ):

    # initialize Q(s, a) and Model(s, a) for all states and actions
    # Model(s, a) is a dictionary of the tuple (R, S')
    Q = np.random.normal(size=(env.state_size, env.action_size))
    # q values for terminals are removed
    for terminal in env.terminals:
        Q[terminal, :] = 0
    # Model is a dictionary with key (s, a) and value (R, S')
    Model = {}

    # Prerequisites
    # epsilon-greedy policy
    def policy(state: int) -> int:
        best_action = np.argmax(Q[state, :])
        dice = np.random.rand()
        if dice < 1-epsilon+epsilon/env.action_size:
            return best_action
        else:
            choices = [c for c in range(env.action_size) if c != best_action]
            choice = np.random.choice(choices)
            return choice

    # the main loop
    for step in tqdm(range(max_steps)):
        state = env.current_state #(a)
        action = policy(state) #(b)
        next_state, reward, terminated = env.step(action) #(c)
        if terminated:
            next_state = env.reset()
        #(d)
        old_q = Q[state, action]
        Q[state, action] = old_q + alpha * (reward + gamma * max(Q[next_state, :]) - old_q)
        #(e)
        Model[(state, action)] = (reward, next_state)
        #(f)
        for k in range(n):
            rand_sa, rand_rs = random.choice(list(Model.items()))
            state, action = rand_sa
            reward, next_state = rand_rs
            old_q = Q[state, action]
            Q[state, action] = old_q + alpha * (reward + gamma * np.max(Q[next_state, :]) - old_q)
    return Q

env = Maze(width=9, height=6, start=(0, 2), goal=(8, 0), 
    walls=[(2, 1), (2, 2), (2, 3), (5, 4), (7, 0), (7, 1), (7, 2)])
Q = tabular_dyna_q(env, max_steps=10000)

# save the table
np.save('tabular_dyna_q.npy', Q)
Q = np.load('tabular_dyna_q.npy')

# Greedy Policy
best_actions = [np.argmax(Q[state, :]) for state in range(env.state_size)]

for y in range(env.height):
    for x in range(env.width):
        state = env._coords_to_state((x, y))
        if state == env.start: print('S', end='')
        elif state not in env.terminals:
            action = best_actions[y*env.width + x]
            if      action==0: print('^', end='')
            elif    action==1: print('>', end='')
            elif    action==2: print('v', end='')
            elif    action==3: print('<', end='')
        else:
            if state == env.goal: print('O', end='')
            else: print('X', end='')
    print()