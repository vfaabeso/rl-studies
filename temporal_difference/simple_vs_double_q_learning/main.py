# comparison on the performance between
# vanilla Q learning and double Q learning
import numpy as np
from tqdm import tqdm
from environment import Environment
from vanilla_q import q_learning
from double_q import double_q_learning

def run() -> None:
    env = Environment()

    q_table = q_learning(env)
    q_table1, q_table2 = double_q_learning(env)

    print('Vanilla Q-table')
    print(q_table)

    print('Double Q-table')
    print(q_table1)
    print(q_table2)

if __name__== '__main__':
    run()