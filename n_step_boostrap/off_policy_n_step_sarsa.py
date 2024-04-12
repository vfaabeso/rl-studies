import numpy as np
from tqdm import tqdm
from gridworld import *
import profile

# define the off policy N-STEP-SARSA algorithm
# return the q table

def off_policy_n_step_sarsa(env) -> list:
    # initialize step size and epsilon
    step_size = 0.01
    # make the target epsilon slightly soft
    # due to importance sampling ratio explosion
    behavior_epsilon, target_epsilon = 0.25, 0.001
    discount = 1
    n = 5 # the n in n-step algorithm
    # episodes
    max_episodes = 500_000

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

    # epsilon greedy policy for both behavior and target policies
    # returns the best action and the probability
    def policy(state: int, epsilon: float) -> tuple:
        dice = np.random.rand()
        # pick the optimal
        best_action, best_value = None, -np.inf
        for action in range(action_size):
            if q_table[(state, action)] > best_value:
                best_action = action
                best_value = q_table[(state, action)]
        # define the probabilities here
        chosen_prob = 1-epsilon+epsilon/action_size
        alter_prob = epsilon/action_size
        if dice < chosen_prob:
            return best_action, chosen_prob
        else: # otherwise, pick randomly
            other_actions = [a for a in range(action_size) if a != best_action]
            alt_action = np.random.choice(other_actions)
            return alt_action, alter_prob

    # loop for each episode
    for episode_idx in tqdm(range(max_episodes)):
        # reset the tuple storage
        # initialize to zero R_0
        state_run, action_run, reward_run = [], [], [0]
        # initialize and store state
        state = env.reset()
        state_run.append(state)
        # choose action from the behavior policy
        action, _ = policy(state, behavior_epsilon)
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
                else: #select and store an action via behavior policy
                    action, _ = policy(next_state, behavior_epsilon)
                    action_run.append(action)
            # define tau
            tau = current_time - n + 1
            # initialize G
            expected = 0
            if tau >= 0:
                # define rho
                rho = 1
                for i in range(tau+1, min(tau+n, episode_length-1)+1):
                    _, target_prob = policy(state_run[i], target_epsilon)
                    _, behavior_prob = policy(state_run[i], behavior_epsilon)
                    ratio = target_prob / behavior_prob
                    rho *= ratio
                # define the value of G
                expected = sum([(discount ** (i - tau - 1)) * reward_run[i]\
                    for i in range(tau+1, min(tau+n, episode_length-1)+1)])
                if tau + n < episode_length:
                    expected = expected + (discount ** n)\
                        * q_table[(state_run[tau+n], action_run[tau+n])]
                old_q = q_table[(state_run[tau], action_run[tau])]
                q_table[(state_run[tau], action_run[tau])] = old_q + step_size * ratio * (expected - old_q)
                # update policy

            current_time += 1
            state = next_state
    # return the table
    return q_table

# init environment
# let's use a smaller environment since
# off policy methods are generally slower to converge
env = GridWorld(width=5, height=5, terminal_coords=(2, 2))
q_table = off_policy_n_step_sarsa(env)

# save the result
np.save('off_policy_n_step_sarsa.npy', q_table)
q_table = np.load('off_policy_n_step_sarsa.npy')
print(q_table)

symbols = '^>v<'
for state in range(env.state_size):
    x, y = env._state_to_coords(state)
    if state == env.terminal:
        print('G', end='')
    else:
        best_action, best_value = None, -np.inf
        for action in range(env.action_size):
            if q_table[(state, action)] > best_value:
                best_action = action
                best_value = q_table[(state, action)]
        print(symbols[best_action], end='')
    if x >= env.width - 1: print()
