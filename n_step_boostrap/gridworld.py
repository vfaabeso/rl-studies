# A sample environment for simulating grid world
# to be used in n step bootstrap
import numpy as np

class GridWorld:
    # terminal coords is (x, y)
    def __init__(self, width=10, height=10, terminal_coords=None):
        self.width = width
        self.height = height
        self.state_size = width * height
        self.action_size = 4
        # this is fixed
        self.terminal = self._set_terminal(terminal_coords)
        self.current_state = self._select_init_state()

    def _set_terminal(self, terminal_coords) -> int:
        if terminal_coords == None:
            return np.random.randint(0, self.state_size)
        else:
            return self._coords_to_state(terminal_coords[0], terminal_coords[1])

    def _select_init_state(self) -> int:
        state = None
        while state == None:
            try_state = np.random.randint(0, self.state_size)
            if try_state != self.terminal:
                state = try_state
        return state

    def _state_to_coords(self, state: int) -> tuple:
        x = state % self.width
        y = state // self.height
        return x, y

    def _coords_to_state(self, x: int, y: int) -> int:
        return y * self.height + x

    def _coords_bounded(self, x: int, y: int) -> bool:
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def reset(self) -> int:
        self.current_state = self._select_init_state()
        return self.current_state

    def possible_actions(self, state: int) -> tuple:
        # check if in terminal
        if state == self.terminal: return ()
        x, y = self._state_to_coords(state)
        possible_actions = []
        # top, right, down, left
        if y > 0: possible_actions.append(0)
        if x < self.width-1: possible_actions.append(1)
        if y < self.height-1: possible_actions.append(2)
        if x > 0: possible_actions.append(3)
        return tuple(possible_actions) 

    # tuple is (new state, reward, is terminated)
    def step(self, action: int) -> tuple:
        x, y = self._state_to_coords(self.current_state)
        if      action==0: y-=1
        elif    action==1: x+=1
        elif    action==2: y+=1
        elif    action==3: x-=1
        # check if on bounds
        if self._coords_bounded(x, y):
            self.current_state = self._coords_to_state(x, y)
            if self.current_state == self.terminal:
                return (self.current_state, 1, True)
            else: return (self.current_state, -1, False)
        return (self.current_state, -1, False)