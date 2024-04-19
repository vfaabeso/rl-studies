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
    theta=1):

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

    # get the state, action pairs that are
    # predicted to lead to a certain state
    def pred_state(state: int) -> list:
        # actions and states to return
        ret_state_act = []
        # get adjacent states
        x, y = env.state_to_coords(state)
        # top, right, down, left
        potential_coords = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
        # contains both the position relative to the state param
        # and the actual state/action (tuple)
        potent_states, potentials = [], []
        
        for idx, coord in enumerate(potential_coords):
            s = env.coords_to_state(coord)
            if s in env.special_cells:
                potent_states.append((idx, s))
        # then get the best actions along with it
        # based on the current q table
        for pstate in potent_states:
            best_act = np.argmax(Q[pstate[1], :])
            best_value = np.max(Q[pstate[1], :])
            potentials.append((pstate[0], pstate[1], best_act, best_value))
        # check if the actions are pointing to the state param
        for psar in potentials:
            p, s, a, r = psa
            if (p + 2) % env.action_size == a:
                ret_state_act.append((s, a, r))
        return ret_state_act

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
            if P > theta: PQueue.push(P, (state, action))
            for _ in range(n):
                # only run if there are elements in it
                if len(PQueue) <= 0: break 
                priority, state_action = PQueue.pop_last()
                state, action = state_action
                reward, next_state = Model[(state, action)]
                old_q = Q[state, action]
                Q[state, action] = old_q + alpha * (reward + gamma * np.max(Q[next_state, :]) - old_q)
                # loop for all S_bar, A_bar predicted to lead to S
                # meaning we refer to all the predicted model
                for state_action, reward_state in list(Model.items()):
                    s, a = state_action
                    r, ns = reward_state
                    if state == ns:
                        P = abs(r + gamma * np.max(Q[state, :]) - Q[s, a])
                        if P > theta: PQueue.push(P, (s, a))

    print(Model)
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