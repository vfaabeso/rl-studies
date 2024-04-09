# Double (Q) Learning demonstration
# using the simple env from Figure 6.5

import numpy as np
from tqdm import tqdm

# There are 4 states and 2 actions
# States: 0-3, 0 and 3 are terminal
# Actions: 0 = Left, 1 = Right
class Environment:
    def __init__(self) -> None:
        self._default_state = 2
        self.current_state = None
        self.state_size = 4
        self.action_size = 2
        self.terminal_states = (0, 3)

    def reset(self) -> int:
        self.current_state = self._default_state
        return self.current_state

    # which actions are possible given the state
    def possible_actions(self, state: int) -> tuple:
        if state == 1: return (0,)
        elif state == 2: return (0, 1)
        else: return ()

    # returns a tuple of (S', R, isTerminated)
    def step(self, action: int) -> tuple:
        if self.current_state == 0:
            # unfounded action
            return (self.current_state, 0, True)
        elif self.current_state == 1:
            if action == 0:
                self.current_state -= 1
                reward = np.random.normal(loc=-0.1)
                return (self.current_state, reward, True)
            # unfounded action
            return (self.current_state, 0, False)
        elif self.current_state == 2:
            if action == 0:
                self.current_state -= 1
                return (self.current_state, 0, False)
            elif action == 1:
                self.current_state += 1
                return (self.current_state, 0, True)
            # unfounded action
            return (self.current_state, 0, False)
        elif self.current_state == 3:
            # unfounded action
            return (self.current_state, 0, True)
        # unfounded action
        else: return (self.current_state, 0, True)

# returns the two q tables
def double_q_learning() -> tuple:
    # initialize step size and epsilon
    step_size = 0.1
    epsilon = 0.1
    discount = 1

    # init environment
    env = Environment()
    # episodes
    max_episodes = 50_000

    # shorthands
    state_size = env.state_size
    action_size = env.action_size
    state_action_size = (state_size, action_size)

    # initialize the two q tables except Q(terminal, *) = 0
    q_table1 = np.random.normal(size=state_action_size)
    for terminal in env.terminal_states:
        for action in range(action_size):
            q_table1[(terminal, action)] = 0
    q_table2 = np.random.normal(size=state_action_size)
    for terminal in env.terminal_states:
        for action in range(action_size):
            q_table2[(terminal, action)] = 0

    # impossible actions should have a q value of zero too
    for state in range(state_size):
        possible_actions = env.possible_actions(state)
        for action in range(action_size):
            if action not in possible_actions:
                q_table1[(state, action)] = 0
                q_table2[(state, action)] = 0

    # the current policy we have
    # which is epsilon greedy
    # change the qualifying best q value to be the sum of the two q tables
    def policy(state: int) -> int:
        possible_actions = env.possible_actions(state)
        try:
            if len(possible_actions) == 0:
                raise Exception('No possible actions.')
            # is it the only action possible
            elif len(possible_actions) == 1:
                return possible_actions[0]
            else: # for more than one action
                dice = np.random.rand()
                # pick the optimal
                best_action, best_value = None, -np.inf
                
                for action in possible_actions:
                    q_sum = q_table1[(state, action)] + q_table2[(state, action)]
                    if q_sum > best_value:
                        best_action = action
                        best_value = q_sum
                
                if dice < 1-epsilon+epsilon/action_size:
                    return best_action
                else: # otherwise, pick randomly
                    other_actions = [a for a in range(action_size) if a != best_action]
                    alt_action = np.random.choice(other_actions)
                    return alt_action
        except Exception as e:
            print(str(e))

    # fetching the best q value
    def best_action_value(q_table: list, state: int) -> (int, float):
        best_action, best_value = None, -np.inf
        possible_actions = env.possible_actions(state)
        for action_test in possible_actions:
            if q_table[(state, action_test)] > best_value:
                best_action = action_test
                best_value = q_table[(state, action_test)]
        # sanity check
        return best_action, best_value

    # loop for each episode
    for episode_idx in tqdm(range(max_episodes)):
        # initialize state
        state = env.reset()
        # loop for each step of episode
        terminated = False
        while not terminated:
            # choose A from S using policy
            action = policy(state)
            # take action A, observe R, S'
            new_state, reward, terminated = env.step(action)
            # prerequisites for updating the q table
            # Q1(S, A)
            old_q1 = q_table1[(state, action)]
            # Q2(S, A)
            old_q2 = q_table2[(state, action)]
            # Q1(S', A)
            new_q1 = q_table1[(new_state, action)]
            # Q2(S', A)
            new_q2 = q_table2[(new_state, action)]
            # the argmax of the q tables
            q1_act, _ = best_action_value(q_table1, new_state)
            q2_act, _ = best_action_value(q_table2, new_state)
            # finally update the table
            # with 0.5 probability
            dice = np.random.rand()
            if dice >= 0.5:
                # only update if best action exists
                if q2_act != None:
                    # Q2(S', argmax of a Q1(S', a))
                    q2_final = q_table2[(new_state, q1_act)]
                    rhs = old_q1 + step_size * (reward + discount * q2_final - old_q1)
                    q_table1[(state, action)] = rhs
            else:
                if q1_act != None:
                    # Q1(S', argmax of a Q2(S', a))
                    q1_final = q_table1[(new_state, q2_act)]
                    rhs = old_q2 + step_size * (reward + discount * q1_final - old_q2)
                    q_table2[(state, action)] = rhs
            # set the new state
            state = new_state

    # return the table
    return (q_table1, q_table2)

q_table1, q_table2 = double_q_learning()

env = Environment()

# fetching the best q value
def best_action_value(env, q_table: list, state: int) -> (int, float):
    best_action, best_value = None, -np.inf
    possible_actions = env.possible_actions(state)
    for action_test in possible_actions:
        if q_table[(state, action_test)] > best_value:
            best_action = action_test
            best_value = q_table[(state, action_test)]
    # sanity check
    return best_action, best_value

# all actions should be (S=2, A=1)
state = env.reset()
for step in range(10):
    action, _ = best_action_value(env, q_table1, state)
    print(state, action)
    state, reward, terminated = env.step(action)
    if terminated:
        state = env.reset()