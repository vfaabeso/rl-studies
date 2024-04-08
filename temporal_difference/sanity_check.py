import numpy as np

q_table = np.load('sarsa.npy')

for state in range(48):
    print(q_table[(state)])