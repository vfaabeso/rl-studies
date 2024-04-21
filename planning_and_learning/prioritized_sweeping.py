# built on top of tabular dyna q learning

import numpy as np
import random
from utils.p_queue import PQueue as PQ
from maze import Maze
from tqdm import tqdm

# the main algorithm
def prioritized_sweeping(
    env: Maze,
    max_episodes: int = 50, n: int = 10,
    epsilon: float = 0.2, alpha: float = 0.1, gamma: float = 0.95,
    theta=5):

    # initialize Q(s, a) and Model(s, a) for all states and actions
    # Model(s, a) is a dictionary of the tuple (R, S')
    Q = np.random.normal(size=(env.state_size, env.action_size))
    # q values for terminals are removed
    for terminal in env.terminals:
        Q[terminal, :] = 0
    # Model is a dictionary with key (s, a) and value (R, S')
    Model = {}
    PQueue = PQ() # the custom pqueue

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

    # episodic loop
    for episode_idx in tqdm(range(max_episodes)):
        terminated = False
        env.reset()
        while not terminated:
            state = env.current_state #(a)
            action = policy(state) #(b)
            next_state, reward, terminated = env.step(action) #(c)
            #(d)
            Model[(state, action)] = (reward, next_state)
            #(e)
            P = abs(reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            #(f)
            if P > theta: PQueue.push_replace(P, (state, action))
            for _ in range(n):
                # only run if there are elements in it
                if len(PQueue) <= 0: break 
                priority, (state, action) = PQueue.pop_last()
                reward, next_state = Model[(state, action)]
                old_q = Q[state, action]
                Q[state, action] = old_q + alpha * (reward + gamma * np.max(Q[next_state, :]) - old_q)
                # loop for all S_bar, A_bar predicted to lead to S
                # meaning we refer to all the predicted model
                # get the state being referred to
                ref_state = state
                for state_action, reward_state in list(Model.items()):
                    state_, action_ = state_action
                    reward_, next_state = reward_state
                    if ref_state == next_state:
                        P = abs(reward_ + gamma * np.max(Q[ref_state, :]) - Q[state_, action_])
                        if P > theta: PQueue.push(P, (state_, action_))

    print(PQueue)
    return Q

env = Maze(width=9, height=6, start=(0, 2), goal=(8, 0), 
    walls=[(2, 1), (2, 2), (2, 3), (5, 4), (7, 0), (7, 1), (7, 2)])
Q = prioritized_sweeping(env, max_episodes=50, n=10)

# save the table
np.save('prioritized_sweeping.npy', Q)
Q = np.load('prioritized_sweeping.npy')

# Greedy Policy
best_actions = [np.argmax(Q[state, :]) for state in range(env.state_size)]

for y in range(env.height):
    for x in range(env.width):
        state = env.coords_to_state((x, y))
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