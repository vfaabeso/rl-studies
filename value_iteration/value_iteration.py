import numpy as np

terminals = (0, 15)
state_width = 4
state_height = 4
state_size = state_width * state_height
action_size = 4

value_table = np.random.normal(size=state_size)
for i in terminals: value_table[i] = 0

termination_rate = 1e-4
discount = 0.95

def transition(state: int, action: int) -> (int, int):
   # stuck if terminal
   if state in terminals: return (state, 0)
   x = state % state_width
   y = state // state_height
   if    action == 0: y -= 1
   elif  action == 1: x += 1
   elif  action == 2: y += 1
   elif  action == 3: x -= 1
   # within bounds
   if (x >= 0 and x < state_width) and (y >= 0 and y < state_height):
      new_state = y*state_height+x
      # did it land to the awards?
      if new_state in terminals: return new_state, 1
      else: return new_state, -1
   else: return state, -1

def value_iteration() -> list[int]:
    while True:
        delta = 0
        for state in range(state_size):
            old_value = value_table[state]
             # summation portion
            best_value, best_action = -np.inf, None
            for action in range(action_size):
                next_state, reward = transition(state, action)
                value = reward + discount * value_table[next_state]
                if value > best_value:
                    best_value = value
                    best_action = action
            value_table[state] = best_value
            delta = max(delta, abs(old_value - value_table[state]))
        if delta < termination_rate: break

    # output the policy
    policy_table = np.full(shape=state_size, fill_value=-1)
    for state in range(state_size):
        # summation portion
        best_value, best_action = -np.inf, None
        for action in range(action_size):
            next_state, reward = transition(state, action)
            value = reward + discount * value_table[next_state]
            if value > best_value:
                best_value = value
                best_action = action
        policy_table[state] = best_action
    return policy_table

policy_table = value_iteration()
print(policy_table)