# test cases
import numpy as np
policy_table = np.load('./policy_results/off_policy_mc.npy')

# usable ace case
print('Usable Ace')
for player_sum in range(21, 11, -1):
    print(player_sum, end=' ')
    for dealer_value in range(1, 11):
        if policy_table[(player_sum, dealer_value, 1)]==1:
            print('█', end='')
        else: print(' ', end='')
    print()
print('   ', end='')
for dealer in ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'X'):
    print(dealer, end='')
print('\n')
# no usable ace case
print('No Usable Ace')
for player_sum in range(21, 11, -1):
    print(player_sum, end=' ')
    for dealer_value in range(1, 11):
        if policy_table[(player_sum, dealer_value, 0)]==1:
            print('█', end='')
        else: print(' ', end='')
    print()
print('   ', end='')
for dealer in ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'X'):
    print(dealer, end='')
print('\n')