from numpy import argmax, random, subtract
import matplotlib.pyplot as plt

# the bandit function
def bandit(arguments: dict[str, any], action: int) -> int:
    return round(random.normal(loc=arguments['stand_means'][action], scale=arguments['stand_dev']))

# perform 1 run and return the reward list
def do_one_run(arguments: dict[str, any]) -> list[int]:
    # Initialize, for a = 1 to k
    q_values = [0 for i in range(arguments['K'])] # action-values
    n_chosen = [0 for i in range(arguments['K'])] # number of times the switch is chosen
    reward_list = []

    for step in range(arguments['max_steps']):
        dice = random.rand()
        action = 0
        if dice >= arguments['epsilon']: action = argmax(q_values)
        else: action = random.randint(0, arguments['K'])
        reward = bandit(arguments, action)
        n_chosen[action] += 1
        old_q = q_values[action]
        q_values[action] = old_q + (1/n_chosen[action])*(reward-old_q)
        reward_list.append(reward)

    return reward_list

# perform multiple runs and get the
# average reward for all those runs
def do_multiple_runs(arguments: dict[str, any]) -> list[float]:
    averages = [0 for i in range(arguments['max_steps'])]
    for run in range(arguments['max_runs']):
        average = do_one_run(arguments)
        # then add to the averages
        averages = averages + (1/(run+1))*subtract(average, averages)
    return averages

# perform one experiment
def perform_experiment(K: int, max_steps: int, max_runs: int, epsilon: float, stand_dev: float, stand_means: list[float]) -> list[float]:
    # encapsulate the arguments here
    arguments = {
        'K': K, 'max_steps': max_steps, 'max_runs': max_runs,
        'epsilon': epsilon, 'stand_dev': stand_dev, 'stand_means': stand_means
    }
    # just call multiple runs
    averages = do_multiple_runs(arguments)
    return averages

# the main part of the experiment
K = 10
max_steps = 250
max_runs = 50
stand_dev = 1

# the standard means of each switch
stand_means = [5 * random.normal() for i in range(K)]

# call the experiment
greedy = perform_experiment(K, max_steps, max_runs, epsilon=0.0, stand_dev=stand_dev, stand_means=stand_means)
epsilon1 = perform_experiment(K, max_steps, max_runs, epsilon=0.01, stand_dev=stand_dev, stand_means=stand_means)
epsilon2 = perform_experiment(K, max_steps, max_runs, epsilon=0.05, stand_dev=stand_dev, stand_means=stand_means)

 # plot here
plt.plot([i+1 for i in range(max_steps)], greedy, label='e=0')
plt.plot([i+1 for i in range(max_steps)], epsilon1, label='e=0.01')
plt.plot([i+1 for i in range(max_steps)], epsilon2, label='e=0.05')
plt.title('Reward per step')
plt.xlabel('Step')
plt.ylabel('Reward obtained')
plt.legend()
plt.show()