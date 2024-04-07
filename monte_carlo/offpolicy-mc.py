# modified from on policy first visit mc

import gymnasium as gym
import numpy as np

def off_policy_monte_carlo() -> list:
    # initialize
    env = gym.make("Blackjack-v1")
    # get the shape of the states
    state_shape = tuple(map(lambda space: space.n, env.observation_space.spaces))
    state_action_shape = tuple(list(state_shape) + [env.action_space.n])
    max_episodes = 1_000_000
    discount = 1 # recommended according to the textbook
    epsilon = 0.8 # to encourage exploration more

    quality_table = np.random.normal(size=state_action_shape)
    # cumulative sum table
    cumulative_table = np.full(shape=state_action_shape, fill_value=0)

    # a common function to use
    def greedy_policy(state: tuple) -> int:
        best_val, best_action = -np.inf, None
        for action in range(env.action_space.n):
            # create flattened tuple
            state_action = tuple(sum([list(state), [action]], []))
            if quality_table[state_action] > best_val:
                best_val = quality_table[state_action]
                best_action = action
        return best_action

    # epsilon soft greedy, return both the action
    # and the probability it took to get there
    def epsilon_soft(state: tuple) -> (int, float):
        best_action = greedy_policy(state)
        # Get the probabilities of all of the actions
        dice = np.random.rand()
        # choose randomly if dice not in favor for the best action
        if dice >= 1-epsilon+(epsilon/env.action_space.n):
            # randomly choose
            alternative_actions = np.array([i for i in range(env.action_space.n) if i != best_action])
            alt_action = np.random.choice(alternative_actions)
            return alt_action, epsilon / env.action_space.n
        # otherwise, proceed with current action
        else: return best_action, 1-epsilon+(epsilon/env.action_space.n)

    # generates an episode with a soft policy
    def generate_soft_episode() -> (list, list, int):
        terminated = False
        current_time = 0
        episode_list = [] #accepts a list of (S, A, R)
        prob_list = [] # a list of b(A|S)
        # choose an initial state and action
        old_state, info = env.reset()
        action, prob = epsilon_soft(old_state)
        while not terminated:
            state, reward, terminated, truncated, info = env.step(action)
            # append to the episode list
            # this is the (S_t, A_t, R_t+1) tuple
            episode_list.append((old_state, action, reward))
            prob_list.append(prob)
            old_state = state
            action, prob = epsilon_soft(old_state) # determine the next action
            current_time += 1
        final_time = current_time-1
        return episode_list, prob_list, final_time

    # choose the maximum from quality table
    # that is, policy(s) <- argmax of a of Q(S, a)
    policy_table = np.empty(shape=state_shape)
    for player_sum in range(state_shape[0]):
        for dealer_value in range(state_shape[1]):
            for use_ace in range(state_shape[2]):
                sample_state = (player_sum, dealer_value, use_ace)
                policy_table[sample_state] = greedy_policy(sample_state)

    # episode generation
    # loop forever (or at maximum episode)
    for episode_idx in range(max_episodes):
        # generate an episode from the soft policy defined earlier
        episode_list, prob_list, final_time = generate_soft_episode()
        expected = 0 # G <- 0
        weight = 1 # W <- 1
        # loop for each step of the episode
        for step in range(final_time, -1, -1):
            current_state, current_action, reward = episode_list[step]
            # create a flattened state-action tuple
            current_state_action = tuple(sum([list(current_state), [current_action]], []))
            # G <- gamma * G + R_t+1
            expected = discount * expected + reward
            # C(S_t, A_t) <- C + W
            cumulative_table[current_state_action] = cumulative_table[current_state_action] + weight
            # Q(S_t, A_t) <- Q + (W/C)[G-Q]
            _tmp = (weight/cumulative_table[current_state_action])*(expected-quality_table[current_state_action])
            quality_table[current_state_action] = quality_table[current_state_action] + _tmp
            # policy at S_t <- argmax of a of Q(S_t, a)
            policy_table[current_state] = greedy_policy(current_state)
            # proceed to next episode if A_t != policy at S_t
            if current_action != policy_table[current_state]: break
            # W <- W * (1/b(A|S))
            weight = weight / prob_list[step]
    return policy_table

# run the simulation
policy_table = off_policy_monte_carlo()
np.save('monte-carlo.npy', policy_table)

# test cases
policy_table = np.load('monte-carlo.npy')

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