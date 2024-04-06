# REMOVES THE ASSUMPTION THAT TAKING AN ACTION AT A GIVEN STATE
# WOULD CERTAINLY MAKES US MOVE TO THE DESIRED STATE
# THAT IS, TRANSITION IS NON-DETERMINISTIC

# IN THIS EXAMPLE, THERE IS A CONSTANT WIND BLOWING WEST
# THEREFORE, THERE'S A CHANCE THAT THE AGENT WOULD MOVE WESTWARD
# IN ITS FINAL POSITION

import numpy as np

terminals = (0, 15)
state_width = 4
state_height = 4
state_size = state_width * state_height
action_size = 4

value_table = np.random.normal(size=state_size)
for i in terminals: value_table[i] = 0
policy_table = np.random.randint(0, action_size, size=state_size)

termination_rate = 1e-4
discount = 0.95
intended_prob = 0.8

# this is modified into transition probabilities
# tuple is modified with the third argument being the probability
def transition(state: int, action: int) -> list[(int, int, float)]:
   # stuck if terminal
   if state in terminals: return [(state, 0, 1.0)]

   # the intended position
   intended_position = None
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
      if new_state in terminals:
         intended_position = (new_state, 1, intended_prob)
      else: intended_position = (new_state, -1, intended_prob)
   else: intended_position = (state, -1, intended_prob)

   # the alternative position affected by wind
   alternative_position = None
   x -= 1 # apply wind
   # check bounds
   if (x >= 0 and x < state_width) and (y >= 0 and y < state_height):
      new_state = y*state_height+x
      # did it land to the awards?
      if new_state in terminals:
         alternative_position = (new_state, 1, 1-intended_prob)
      else: alternative_position = (new_state, -1, 1-intended_prob)
   else: alternative_position = (state, -1, 1-intended_prob)
   # return possible transitions
   return [intended_position, alternative_position]

def policy_evaluation() -> None:
   while True:
      delta = 0
      for state in range(state_size):
         value = value_table[state]
         # the summation portion
         action = policy_table[state]
         transitions = transition(state, action)
         value_sum = 0
         # taking account transitions
         for next_state, reward, prob in transitions:
            expected = reward + discount * value_table[next_state]
            product = prob * expected
            value_sum += product
         value_table[state] = value_sum
         delta = max(delta, abs(value - value_table[state]))
      if delta < termination_rate: break

def policy_improvement() -> bool:
   policy_stable = True
   for state in range(state_size):
      old_action = policy_table[state]
      # summation portion
      best_value, best_action = -np.inf, None
      for action in range(action_size):
         transitions = transition(state, action)
         value_sum = 0
         # taking account all transitions
         for next_state, reward, prob in transitions:
            expected = reward + discount * value_table[next_state]
            product = prob * expected
            value_sum += product
         if value_sum > best_value:
            best_value = value_sum
            best_action = action
      policy_table[state] = best_action
      if old_action != policy_table[state]: policy_stable = False
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
print(policy_table)
