import numpy as np
from tqdm import tqdm
from environment import Environment

# just return one q table,
def q_learning(env: Environment) -> list:
    # initialize step size and epsilon
    step_size = 0.01
    epsilon = 0.1
    discount = 1

    # episodes
    max_episodes = 50_000

    # shorthands
    state_size = env.state_size
    action_size = env.action_size
    state_action_size = (state_size, action_size)

    # initialize the q table except Q(terminal, *) = 0
    q_table = np.random.normal(size=state_action_size)
    for terminal in env.terminal_states:
        for action in range(action_size):
            q_table[(terminal, action)] = 0

    # impossible actions should have a q value of zero too
    for state in range(state_size):
        possible_actions = env.possible_actions(state)
        for action in range(action_size):
            if action not in possible_actions:
                q_table[(state, action)] = 0

    # the current policy we have
    # which is epsilon greedy
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
                    if q_table[(state, action)] > best_value:
                        best_action = action
                        best_value = q_table[(state, action)]
                
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
            old_table = q_table[(state, action)]
            # for computing max a of Q(S', a)
            best_act, best_q = best_action_value(q_table, new_state)
            # finally update the table
            if best_act != None:
                rhs = old_table + step_size * (reward + discount * best_q - old_table)
                q_table[(state, action)] = rhs
            # set the new state
            state = new_state

    # return the table
    return q_table