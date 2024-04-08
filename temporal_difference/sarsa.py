import gymnasium as gym
import numpy as np

# define the SARSA algorithm
# return the q table
def sarsa() -> list:
    # initialize step size and epsilon
    step_size = 0.01
    epsilon = 0.2
    discount = 1

    # init environment
    env = gym.make("CliffWalking-v0")
    # episodes
    max_episodes = 1_000_000

    # shorthands
    state_size = env.observation_space.n
    action_size = env.action_space.n
    state_action_size = (state_size, action_size)

    # initialize state-action table except Q(terminal, *) = 0
    q_table = np.random.normal(size=state_action_size)
    # terminals are the cliff states and the goal state
    # set the q value for terminals to zero
    # terminals are states 37-47
    for terminal in range(37, 48):
        for action in range(action_size):
            q_table[(terminal, action)] = 0

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
    for episode_idx in range(max_episodes):
        # initialize state
        state, _ = env.reset()
        # choose action from the current policy
        action = policy(state)
        # loop for each step of episode
        terminated = False
        while not terminated:
            # take action A, observe R, S'
            new_state, reward, terminated, _, _ = env.step(action)
            # choose A' from S' using policy
            new_action = policy(new_state)
            # update q table
            old_table = q_table[(state, action)]
            new_table = q_table[(new_state, new_action)]
            update = old_table + step_size * (reward + discount * new_table - old_table)
            q_table[(state, action)] = update
            # set the new state and action
            state = new_state; action = new_action

    # return the table
    return q_table

#test the sarsa algorithm here
env = gym.make("CliffWalking-v0", render_mode="human")
q_table = sarsa()

# use the greedy approach instead
def greedy_policy(q_table: list, state: int) -> int:
    best_action, best_value = None, -np.inf
    for action in range(env.action_space.n):
        if q_table[(state, action)] > best_value:
            best_action = action
            best_value = q_table[(state, action)]
    return best_action

observation, info = env.reset()
for _ in range(1000):
   action = greedy_policy(q_table, observation)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()