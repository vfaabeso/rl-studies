import gymnasium as gym
import numpy as np

env = gym.make("Blackjack-v1")
samples = 1_000_000

array_shape = (32, 11, 2)
count = np.full(shape=array_shape, fill_value=0)

for sample in range(samples):
    state, info = env.reset()
    count[state] += 1
    terminated = False

    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        count[state] += 1    

# plot the data
import matplotlib.pyplot as plt

x = np.arange(array_shape[1])
y = np.arange(array_shape[0])
X, Y = np.meshgrid(x, y)
Z = np.sum(count, axis=2)

plt.figure()
plt.pcolormesh(X, Y, Z, cmap='viridis', edgecolors='k', linewidths=0.5)
plt.colorbar(label='Z')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()