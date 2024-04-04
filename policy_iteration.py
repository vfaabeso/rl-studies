import numpy as np

# hyperparameters
theta = 1e-6
discount = 0.90

# environment functions
state_width = 4
state_height = 4
state_size = state_width * state_height
action_size = 4
terminal = (0,15)

# necessary variables
state_value_table = np.random.randn(state_size)
# set terminal values
for index in range(state_value_table.size):
    if index in terminal: state_value_table[index] = 0
# what to do per state
policy_table = np.random.randint(0, action_size, size=(state_size,))

# helper functions
def _state_to_coords(state: int) -> (int, int):
    x = state % state_width
    y = state // state_height
    return x, y

def _coord_to_state(x: int, y: int) -> int:
    return y*state_height + x

def _is_valid_state(state: int) -> bool:
    return state >= 0 and state < state_size

def _get_adjacent_states(state: int) -> list[int]:
    # get the four directions
    x, y = _state_to_coords(state)
    # top, right, down, left in order
    adjacent_list = [
        _coord_to_state(x, y-1),
        _coord_to_state(x+1, y),
        _coord_to_state(x, y+1),
        _coord_to_state(x-1, y),
    ]
    # filter if valid
    adjacent_list = list(filter(lambda state: _is_valid_state(state), adjacent_list))
    return adjacent_list

def get_reward(state: int) -> int:
    if state in terminal: return 1
    else: return -1

def get_policy(state: int) -> int:
    return policy_table[state]

def get_transition_prob(state_from: int, state_to: int) -> float:
    adjacent_states = _get_adjacent_states(state_from)
    if state_to in adjacent_states: return 1/len(adjacent_states)
    else: return 0

def get_next_state(state: int, action: int) -> int:
    x, y = _state_to_coords(state)
    if action == 0: y -= 1
    elif action == 1: x += 1
    elif action == 2: y += 1
    elif action == 3: x -= 1
    # if invalid, then stay
    new_state = _coord_to_state(x, y)
    if not _is_valid_state(new_state): return state
    else: return new_state

# policy evaluation
def policy_evaluation() -> None:
    delta = 0
    while True:
        for state in range(state_size):
            value = state_value_table[state]
            # since there's just one possible next state
            policy_action = get_policy(state)
            next_state = get_next_state(state, policy_action)
            bellman = get_reward(next_state) + discount*state_value_table[next_state]
            state_value_table[next_state] = bellman
            delta = abs(value - state_value_table[next_state])
            if delta < theta: return

# policy improvement
def policy_improvement() -> bool:
    policy_stable = True
    for state in range(state_size):
        old_action = get_policy(state)
        # define the bellman equation here
        action_values = {}
        for action in range(action_size):
            # since there's one possible next state
            next_state = get_next_state(state, action)
            expected_reward = get_reward(next_state) + discount*state_value_table[next_state]
            action_values[action] = expected_reward
        # then get which is the maximum
        argmax = max(action_values, key=action_values.get)
        policy_table[state] = argmax
        if old_action != argmax: policy_stable = False
    return policy_stable

# loop
policy_stable = False
iteration = 0
while not policy_stable:
    print(f'Iteration #{iteration}')
    policy_evaluation()
    policy_stable = policy_improvement()
    iteration += 1
print(policy_table)
print(state_value_table)
