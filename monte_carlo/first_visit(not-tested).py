import gymnasium as gym
import numpy as np
env = gym.make("Blackjack-v1")

# get the shape of the states
state_shape = tuple(map(lambda space: space.n, env.observation_space.spaces))
# the policy space has the axes
# (player's total sum, dealer showing card value, usable ace)
# which has the size (32, 11, 2)
# policy(obs) -> 0 or 1
policy_table = np.random.randint(0, env.action_space.n, size=state_shape)

max_episodes = 500_000
discount = 1 # recommended according to the textbook

def first_visit_monte_carlo() -> None:
    # initialize
    # V(s) is an element of Reals arbitrarily, for all states
    value_table = np.random.normal(size=state_shape)
    # empty list per possible state
    returns_table = np.full(shape=state_shape, fill_value=[])

    # episode generation
    # loop forever (or at maximum episode)
    for episode_idx in range(max_episodes):
        terminated = False
        current_time = 0
        # generate episode via the current policy
        episode_list = [] #accepts a tuple of (S, A, R)
        old_state, info = env.reset(seed=42)
        while not terminated:
            action = policy_table[state]
            state, reward, terminated, truncated, info = env.step(action)
            # append to the episode list
            episode_list.append((old_state, action, reward))
            old_state = state
            current_time += 1
        final_time = current_time - 1
        # G <- 0
        expected = 0
        # loop for each step of the episode
        visited_states = [] # keep track of visited states
        for step in range(final_time-1, 0):
            reward = episode_list[step+1][2]
            # G <- gamma * G + R_t+1
            expected = discount * expected + reward
            # unless S_t appears in S_0, S_1, ..., S_t-1
            current_state = episode_list[step][0]
            if current_state not in visited_states:
                # append G to Returns(S_t)
                returns_table[current_state].append(expected)
                # V(S_t) <- Average(Returns(S_t))
                value_table[current_state] = np.average(returns_table[current_state])
    
# env = gym.make("Blackjack-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()