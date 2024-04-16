class Maze:
    def __init__(self, width: int, height: int, start: tuple, goal: tuple, walls: list):
        self.width = width
        self.height = height
        # these are expressed in discrete states
        self.start = self._coords_to_state(start)
        self.goal = self._coords_to_state(goal)
        self.walls = [self._coords_to_state(c) for c in walls]
        self.terminals = self.walls + [self.goal]
        # native to every environment
        self.current_state = self.start
        self.state_size = width * height
        self.action_size = 4

    def reset(self) -> int:
        self.current_state = self.start
        return self.current_state

    # return the next state, reward, and if terminated
    # Action: 0=top, 1=right, 2=down, 3=left
    def step(self, action: int) -> tuple:
        x, y = self._state_to_coords(self.current_state)
        if      action==0: y-=1
        elif    action==1: x+=1
        elif    action==2: y+=1
        elif    action==3: x-=1
        # check the new x and y
        # if within bounds
        if x>=0 and x<self.width and y>=0 and y<self.height:
            # if hit the wall
            new_state = self._coords_to_state((x, y))
            if new_state in self.walls:
                return (self.current_state, 0, False)
            # if on the goal
            elif new_state == self.goal:
                self.current_state = new_state
                return (self.current_state, 1, True)
            # just in bounds
            else:
                self.current_state = new_state
                return (self.current_state, 0, False)
        # going out of bounds
        else: return (self.current_state, 0, False)

    # helper functions
    def _coords_to_state(self, coords: tuple) -> int:
        return coords[1] * self.width + coords[0]

    def _state_to_coords(self, state: int) -> tuple:
        x = state % self.width
        y = state // self.width
        return (x, y)
