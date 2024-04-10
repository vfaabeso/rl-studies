# A sample environment for simulating grid world
import numpy as np

class GridWorld:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.state_size = width * height
        self.action_size = 4
        self.terminal = np.random.randint(0, self.state_size)
        self.current_state = self._select_init_state()

    def _select_init_state(self) -> int:
        state = None
        while state == None:
            try_state = np.random.randint(0, self.state_size)
            if try_state != self.terminal:
                state = try_state
        return state

    def reset(self) -> int:
        self.current_state = self._select_init_state()
        return self.current_state
    
    def possible_actions(self, state: int) -> tuple:
        raise NotImplementedError

    def step(self, action: int) -> tuple:
        raise NotImplementedError