import gymnasium as gym
import numpy as np
from tqdm import tqdm

def monte_carlo_exploring_starts() -> list:
    # initialize
    env = gym.make("Blackjack-v1")
    # get the shape of the states
    state_shape = tuple(map(lambda space: space.n, env.observation_space.spaces))
    state_action_shape = tuple(list(state_shape) + [env.action_space.n])
    max_episodes = 10_000_000
    discount = 1 # recommended according to the textbook

    # the policy space has the axes
    # (player's total sum, dealer showing card value, usable ace)
    # which has the size (32, 11, 2)
    # policy(obs) -> 0 or 1
    policy_table = np.random.randint(0, env.action_space.n, size=state_shape)
    # q values
    quality_table = np.random.normal(size=state_action_shape)
    # # empty list per possible state
    # returns_table = np.empty(shape=state_action_shape, dtype=object)
    # returns_table.fill([])
    # how many times it has been visited instead of making a list table above
    returns_count = np.full(shape=state_action_shape, fill_value=0)

    # episode generation
    # loop forever (or at maximum episode)
    for episode_idx in tqdm(range(max_episodes)):
        #print(episode_idx)
        terminated = False
        current_time = 0
        # choose an initial state and action
        old_state, info = env.reset()
        action = env.action_space.sample()
        # generate episode via the current policy
        episode_list = [] #accepts a list of (S, A, R)
        while not terminated:
            state, reward, terminated, truncated, info = env.step(action)
            # append to the episode list
            # this is the (S_t, A_t, R_t+1) tuple
            episode_list.append((old_state, action, reward))
            old_state = state
            action = policy_table[old_state] # determine the next action
            current_time += 1
        final_time = current_time-1
        # G <- 0
        expected = 0
        # loop for each step of the episode
        visited_pairs = [] # keep track of visited pair of states and actions
        for step in range(final_time, -1, -1):
            current_state, current_action, reward = episode_list[step]
            # G <- gamma * G + R_t+1
            expected = discount * expected + reward
            # unless the state-action pair appears in the visited pairs
            if (current_state, current_action) not in visited_pairs:
                current_state_action = (current_state, current_action)
                # append in visited pair
                visited_pairs.append(current_state_action)
                # flatten current state action
                flat_state_action = tuple(sum([list(current_state), [current_action]], []))
                # append G to Returns(S_t)
                # this is not necessary anymore since we're recomputing averages
                # (returns_table[flat_state_action]).append(expected)
                # Q(S_t, A_t) <- average(Returns(S_t, A_t))
                # we'll recompute the averages here
                old_average = quality_table[flat_state_action]
                # recalculate count
                returns_count[flat_state_action] += 1
                new_count = returns_count[flat_state_action]
                new_average = old_average + (1/new_count) * (expected - old_average)
                # quality_table[flat_state_action] = np.average(returns_table[flat_state_action])
                quality_table[flat_state_action] = new_average
                # policy(S_t) <- argmax of a (Q(S_t, a))
                best_action, best_value = None, -np.inf
                for test_action in range(env.action_space.n):
                    flat_test_action = tuple(sum([list(current_state), [test_action]], []))
                    if quality_table[flat_test_action] > best_value:
                        best_action = test_action
                        best_value = quality_table[flat_test_action]
                policy_table[current_state] = best_action
    return policy_table

# run the simulation
policy_table = monte_carlo_exploring_starts()
np.save('mc_exploring_starts.npy', policy_table)

# # test if it wins constantly
# env = gym.make("Blackjack-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = policy_table[observation]  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         # get the end result
#         print(observation, reward)
#         observation, info = env.reset()

# env.close()