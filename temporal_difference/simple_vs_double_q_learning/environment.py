import numpy as np

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