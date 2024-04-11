import numpy as np
from tqdm import tqdm
from gridworld import *

# define the N-STEP-SARSA algorithm
# return the q table
def sarsa() -> list:
    # initialize step size and epsilon
    step_size = 0.01
    epsilon = 0.1
    discount = 1
    n = 10 # the n in n-step algorithm

    # init environment
    # with default width and height of 10
    env = GridWorld()
    # episodes
    max_episodes = 50_000

    # shorthands
    state_size = env.state_size
    action_size = env.action_size
    state_action_size = (state_size, action_size)

    # initialize state-action table except Q(terminal, *) = 0
    q_table = np.random.normal(size=state_action_size)
    # the only terminal is the goal
    for action in range(action_size):
        q_table[(env.terminal, action)] = 0

    # this is where we store the episode information
    # of tuples (S_t, A_t, R_t+1)
    state_run, action_run, reward_run = [], [], []

    # the current policy we have
    # which is epsilon greedy
    def policy(state: int) -> int:
        dice = np.random.rand()
        # pick the optimal
        best_action, best_value = None, -np.inf
        for action in range(action_size):
            if q_table[(state, action)] > best_value:
                best_action = action
                best_value = q_table[(state, action)]
        if dice < 1-epsilon+epsilon/action_size:
            return best_action
        else: # otherwise, pick randomly
            other_actions = [a for a in range(action_size) if a != best_action]
            alt_action = np.random.choice(other_actions)
            return alt_action

    # loop for each episode
    for episode_idx in tqdm(range(max_episodes)):
        # reset the tuple storage
        # initialize to zero R_0
        state_run, action_run, reward_run = [], [], [0]
        # initialize and store state
        state = env.reset()
        state_run.append(state)
        # choose action from the current policy
        action = policy(state)
        action_run.append(action)
        # loop for each step of episode
        terminated = False
        # T = inf
        episode_length = np.inf
        current_time = 0
        # loop for all t
        while not terminated:
            if current_time < episode_length:
                # take action and observe next vars
                next_state, reward, terminated = env.step(action)
                # store the next state and reward
                state_run.append(next_state)
                reward_run.append(reward)
                # if S_t+1 is terminated
                if terminated:
                    episode_length = current_time + 1
                else:
                    action = policy(next_state)
            # define tau
            tau = current_time - n + 1
            # initialize G
            expected = 0
            if tau >= 0:
                # define the value of G
                expected = sum((discount ** (i - tau - 1)) * reward_run[i]\
                    for i in range(tau+1, min(tau+n, episode_length)))
                if tau + n < episode_length:
                    expected = expected + (discount ** n)\
                        * q_table[(state_run[tau+n], action_run[tau+n])]
                old_q = q_table[(state_run[tau], action_run[tau])]
                q_table[(state_run[tau], action_run[tau])] = old_q + step_size * (expected - old_q)
                # update policy (already implied)
            current_time += 1

    # return the table
    return q_table

