import pandas as pd

'''
convert a basic strategy learned Q-table produced by reinforcement learning to a basic strategy table
in a Q-table, rows are states, columns are actions, and values are expected rewards
in a basic strategy table, rows are player hands, columns are dealer upcards, and values are strings representing the best move
'''


orig_table = pd.read_csv('basic_strategy.csv',index_col=0)

q_table = pd.read_csv('bj_q_table_bs.csv',index_col=0)

new_table = orig_table.applymap(lambda x: ' ')
new_table.loc['4'] = 'hit'

for index in q_table.index:
    upcard = index.split(':')[1]
    upcard = upcard[1:upcard.index('hand')-1]
    hand = index.split(':')[2]
    hand = hand[1:]
    if ' ' in hand:
        hand = hand.split(' ')[0] + hand.split(' ')[1]
    if hand in ['JJ','QQ','KK']:
        hand = '1010'
    if 'A' in hand and hand[1:].isnumeric() and int(hand[1:]) > 10:
        hand = str(int(hand[1:])+1)
    q_row = q_table.loc[index,:]
    q_row = q_row[q_row!=0]
    action = q_row.index[q_row==q_row.max()]
    new_table.loc[hand,upcard] = action[0]

new_table.to_csv('q_basic_strategy.csv')
