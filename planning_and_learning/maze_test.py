from maze import Maze

env = Maze(width=9, height=6, start=(0, 2), goal=(8, 0), 
    walls=[(2, 1), (2, 2), (2, 3), (5, 4), (7, 0), (7, 1), (7, 2)])

for action in (0,0,0):
    state, _, _ = env.step(action)
    