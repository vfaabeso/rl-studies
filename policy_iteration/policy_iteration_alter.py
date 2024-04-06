# ALTERNATIVE FORMULATION OF POLICY EVALUATION
# WHERE ACTIONS ARE PROBABILISTIC
import numpy as np

terminals = (0, 15)
state_width = 4
state_height = 4
state_size = state_width * state_height
action_size = 4

value_table = np.random.normal(size=state_size)
for i in terminals: value_table[i] = 0
# determines the chance of getting picked
policy_tally = np.full(shape=(state_size, action_size), fill_value=1)

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

def policy_evaluation() -> None:
   while True:
      delta = 0
      for state in range(state_size):
         value = value_table[state]
         # the summation portion
         value_sum = 0
         # use the probabilities here
         action_probabilities = policy_tally[state] / np.sum(policy_tally[state])
         for action in range(action_size):
            probability = action_probabilities[action]
            next_state, reward = transition(state, action)
            expected = reward + discount * value_table[next_state]
            product = probability * expected
            value_sum += product
         value_table[state] = value_sum
         delta = max(delta, abs(value - value_table[state]))
      if delta < termination_rate: break

def policy_improvement() -> bool:
   policy_stable = True
   # to not be created every time
   action_list = np.arange(action_size)
   for state in range(state_size):
      # pick randomly
      action_probabilities = policy_tally[state] / np.sum(policy_tally[state])
      old_action = np.random.choice(action_list, p=action_probabilities)
      # summation portion
      best_value, best_action = -np.inf, None
      for action in range(action_size):
         next_state, reward = transition(state, action)
         value = reward + discount * value_table[next_state]
         if value > best_value:
            best_value = value
            best_action = action
      policy_tally[state][best_action] += 1
      # pick an action again
      action_probabilities = policy_tally[state] / np.sum(policy_tally[state])
      new_action = np.random.choice(action_list, p=action_probabilities)
      if old_action != new_action: policy_stable = False
   return policy_stable

def policy_iteration() -> None:
   iters = 1
   while True:
      print(f'Iter #{iters}')
      policy_evaluation()
      policy_stable = policy_improvement()
      if policy_stable: break
      iters += 1

policy_iteration()
print(value_table)
print(policy_tally)
