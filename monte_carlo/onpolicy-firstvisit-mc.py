# for epsilon-soft policies
# this is modified version of monte carlo ES

import gymnasium as gym
import numpy as np

def monte_carlo_exploring_starts() -> list:
    # initialize
    env = gym.make("Blackjack-v1")
    # get the shape of the states
    state_shape = tuple(map(lambda space: space.n, env.observation_space.spaces))
    state_action_shape = tuple(list(state_shape) + [env.action_space.n])
    max_episodes = 1_000_000
    discount = 1 # recommended according to the textbook
    epsilon = 0.20 # since we're doing an epsilon-soft policy 
    # higher epsilon means more random choosing is allowed

    policy_table = np.random.randint(0, env.action_space.n, size=state_shape)
    quality_table = np.random.normal(size=state_action_shape)
    returns_count = np.full(shape=state_action_shape, fill_value=0)

    # episode generation
    # loop forever (or at maximum episode)
    for episode_idx in range(max_episodes):
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
                # we'll recompute the averages here
                old_average = quality_table[flat_state_action]
                # recalculate count
                returns_count[flat_state_action] += 1
                new_count = returns_count[flat_state_action]
                new_average = old_average + (1/new_count) * (expected - old_average)
                # Q(S_t, A_t) <- average (Returns(S_t, A_t))
                quality_table[flat_state_action] = new_average
                # A* <- argmax of a (Q(S_t, a))
                best_action, best_value = None, -np.inf
                for test_action in range(env.action_space.n):
                    flat_test_action = tuple(sum([list(current_state), [test_action]], []))
                    if quality_table[flat_test_action] > best_value:
                        best_action = test_action
                        best_value = quality_table[flat_test_action]
                ########################################################
                # MODIFIED SECTION
                # For all possible actions in a given state:
                # Get the probabilities of all of the actions
                dice = np.random.rand()
                # choose randomly if dice not in favor for the best action
                if dice >= 1-epsilon+(epsilon/env.action_space.n):
                    # randomly choose
                    alternative_actions = np.array([i for i in range(env.action_space.n) if i != best_action])
                    alt_action = np.random.choice(alternative_actions)
                    policy_table[current_state] = best_action
                # otherwise, proceed with current action
                else: policy_table[current_state] = best_action
    return policy_table

# run the simulation
policy_table = monte_carlo_exploring_starts()
np.save('policy_table_epsilon_soft.npy', policy_table)

# test cases
policy_table = np.load('policy_table_epsilon_soft.npy')

# usable ace case
print('Usable Ace')
for player_sum in range(21, 10, -1):
    print(player_sum, end=' ')
    for dealer_value in range(11):
        if policy_table[(player_sum, dealer_value, 1)]==1:
            print('█', end='')
        else: print(' ', end='')
    print()
print('   ', end='')
for dealer in ('0', 'A', '2', '3', '4', '5', '6', '7', '8', '9', 'X'):
    print(dealer, end='')
print('\n')
# no usable ace case
print('No Usable Ace')
for player_sum in range(21, 10, -1):
    print(player_sum, end=' ')
    for dealer_value in range(11):
        if policy_table[(player_sum, dealer_value, 0)]==1:
            print('█', end='')
        else: print(' ', end='')
    print()
print('   ', end='')
for dealer in ('0', 'A', '2', '3', '4', '5', '6', '7', '8', '9', 'X'):
    print(dealer, end='')
print('\n')

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