import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import multiprocessing

'''
Reinforcement learning example using tic tac toe

Action selection uses the tabular Q-value method and Upper Confidence Bound
Q-table is a Pandas dataframe where:
    columns are actions in (r,c) format corresponding to spot on the tic-tac-toe board
    rows are states, and are 2D arrays of size 3x3 representing the tic-tac-toe board
    values are tuples of (q,t) where q is the action-state value and t is the number of times the action has been chosen
    
Action-state values are determined by averaging the rewards (-1 for loss, 1 for win) for each game where that action-state combination occured

Our opponent plays according to a simple, stationary set of logic:
    probability of playing on any empty given space is equal,
    except if the space is adjacent (including diagonally) to any spaces it already claimed, in which case the probability doubles
'''

# the row indexes in our Q-table will be versions of the board; pandas doesn't like using 2D lists as indexes so we create a wrapper class and use the hash as the index
class Board:
    def __init__(self,board):
        self.board=board
    def __hash__(self):
        return hash(tuple(self.board[0]+self.board[1]+self.board[2])) # python won't hash lists
    def __eq__(self,other):
        return self.board==other.board
    
# runs through one learning iteration of tf games
def learn_game(c=1,tf=1000): # parameters: c- degree of exploration in Upper Confidence Bound action selection; tf- number of games played to learn
    def init_q(): # initialize Q table with actions, but no states
        actions = []
        for r in range(3):
            for c in range(3):
                actions.append((r,c))
        return pd.DataFrame(columns=actions)

    def UCB(q,n,t):
        if n==0: n=0.0000000001 # since we can't have 0 in the denominator, set an arbitrarily small value so that this value is guaranteed to be the highest
        return q + c*(2*math.log(t)/n)**0.5 # return action-state value adjusted by confidence interval

    # use the board state and q_table to select an action
    def player_move(board,q_table,t):
        iboard = Board(board) # wrapper class used as q-table index
        if hash(iboard) not in q_table.index: # if we haven't seen this state yet, set each (q,t) pair to (0,0) for that state
            indexes = list(q_table.index) # doing the append removes the index names, so we have to save them
            q_table = q_table.append(pd.Series([(0,0)]*9,index=q_table.columns),ignore_index=True)
            q_table.index = indexes+[hash(iboard)]
            for r in range(3):
                for c in range(3):
                    if board[r][c] != ' ':
                        q_table.loc[hash(iboard)][(r,c)]== np.NaN # if there's a letter on a spot on the board, make the corresponding action-state value NaN
            return np.random.choice(q_table.loc[hash(iboard)][:].dropna().index),q_table # pick a random (r,c) from empty columns, since we have no q-values
        
        # we have to use upper confidence bound action selection to chose our action
        options = q_table.loc[hash(iboard)][:].dropna() # select our (r,c) from ones that don't already have a letter (value isn't np.NaN)
        for i,qn in enumerate(options): # for index and action-state value + number of occurrences tuple
            q,n = qn # q-value and number of times used for each action-state
            options.iloc[i] = UCB(q,n,t) # replace q-values with UCB-adjusted values
        argmax = options.max() # find the largest value
        actions = list(options[options==argmax].index) # get the (r,c) action(s) where the UCB value is greatest
        i = np.random.randint(len(actions)) # if more than one action, arbitrarily pick, else return the action
        return actions[i],q_table
                        
    # have the non-RL player make a move
    def other_move(board,actions):
        probs = [[1 for c in range(3)] for r in range(3)] # all spots on the board initialize with the same probability
        for r in range(3):
            for c in range(3):
                if board[r][c] in ['X','O']: probs[r][c] = 0 # not going to move there if there's already a letter
                # if the spot is adjacent to another of the player's marks, the spot is twice as likely to be picked
                elif ((r-1>0 and c-1>0 and board[r-1][c-1]=='O') or (r+1<3 and c-1>0 and board[r+1][c-1]=='O') or (r+1<3 and c+1<3 and board[r+1][c+1]=='O') or (r-1>0 and c+1<3 and board[r-1][c+1]=='O')):
                      probs[r][c] = 2
        onedim_probs = np.array(probs[0]+probs[1]+probs[2])/np.sum(probs[0]+probs[1]+probs[2]) # flatten the probability array and make its sum equal 1.0
        action = np.random.choice(actions,p=onedim_probs) # chose a spot based on the probability distribution
        return action

    # check for winner
    def check_winner(board):
        def check_h(p): # check horizontally for 3 p's in a row
            for r in range(3):
                if board[r] == [p]*3:
                    return True
            return False
        def check_v(p): # check vertically for 3 p's in a row
            for c in range(3):
                if board[:][c] == [p]*3:
                    return True
            return False
        def check_d(p): # check diagonally for 3 p's in a row
            if board[0][0]+board[1][1]+board[2][2]==p*3 or board[0][2]+board[1][1]+board[2][0]==p*3:
                return True
            return False

        # check win for each player
        for player in ['X','O']:
            if check_h(player)+check_v(player)+check_d(player): return player
        for r in range(3):
            if ' ' in board[r]:
                return None
        return 'tie'
            

    q_table = init_q()
    board = [[' ' for c in range(3)] for r in range(3)]
    results = [] # append wins and losses here
    
    
    for t in range(1,tf+1): # play tf number of games
        as_pairs = [] # keep track of all actions-states chosen in form (action,state)
        board = [[' ' for c in range(3)] for r in range(3)]
        
        while True: # loop for moves inside game
            action,q_table = player_move(board,q_table,t) # get action from q-table using UCB
            as_pairs.append((action,hash(Board(board)))) # add this transition to our list (we have to wrap up the board in its hash)
            r = action[0];c = action[1]
            board[r][c] = 'X' # put our X down where chosen

            # if our RL model won, add a win to results and end the game loop
            if check_winner(board) == 'X': results.append(1);break
            elif check_winner(board) == 'tie': results.append(0);break

            # now the "stupid" model plays
            r,c = other_move(board,q_table.columns)
            board[r][c] = 'O'
            # if we lost, add a loss and end game loop
            if check_winner(board) == 'O': results.append(-1);break
            elif check_winner(board) == 'tie': results.append(0);break


        # now we adjust each of the action-state steps taken according to the result of the game
        result = results[-1]
        for a,s in as_pairs: # for each action-state transition from the last game...
            q,n = q_table.loc[s][a] # get the old q-n pair
            q_table.loc[s][a] = ((q*n+result)/(n+1),n+1) # update the q-table with the new average, and increase the n value by 1

    # after all games are played, return the list of wins and losses
    return results

# function that is used by a Process to perform iterations of the learn
def learn_job(output,times,tf):
    arr = []
    for t in range(times):
        arr.append(learn_game(tf=tf))
    output[:] = list(np.mean(np.array(arr),axis=0)) # our output will be the average of all the iterations run
            
if __name__ == '__main__':
    length = 20000 # length of learning curve (number of games per learn)
    scores=[] # will end up being array with values [-1,1] representing average result of nth game in the learning process


    o1 = multiprocessing.Array('d',length) # need to use shared memory variable
    p1 = multiprocessing.Process(target=learn_job,args=(o1,40,length)) # create the process
    p1.start()

    o2 = multiprocessing.Array('d',length) # need to use shared memory variable
    p2 = multiprocessing.Process(target=learn_job,args=(o2,40,length)) # create the process
    p2.start() # start the process

    o3 = multiprocessing.Array('d',length) # need to use shared memory variable
    p3 = multiprocessing.Process(target=learn_job,args=(o3,40,length)) # create the process
    p3.start() # start the process
    
    p1.join();p2.join();p3.join() # wait for all the processes to finish
    
    scores = np.mean(np.array([o1,o2,o3]),axis=0) # average the values from the 3 processes

    ax = plt.gca()
    ax.set_title('Learning Curve for Tic Tac Toe Games, Avg. of 120 Learns')
    ax.set_xlabel('Number of games played')
    ax.set_ylabel('Performance')
    ax.plot(scores)
    plt.show()
 


'''
for x in range(100):
    scores.append(learn_game(tf=8000))
scores=np.array(scores)
mean_scores = np.mean(scores,axis=0)
print(mean_scores)
'''


    
