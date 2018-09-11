import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import multiprocessing
import cards
import blackjack_object as bj

'''
This script contains a function that trains a reinforcement learning model to play blackjack using Q-learning (this version DOES NOT use CV bins as part of the state set; in effect, it is a basic strategy model)
'''
    
# runs through one learning iteration of tf games
def learn_game(c=1,tf=1000,rules=None): # parameters: c- degree of exploration in Upper Confidence Bound action selection; tf- number of games played to learn
    '''
    Uses Q-learning to generate a Q-table of action-state values over many iterations of blackjack hands

    paramters:
        c: the exploration term coefficient in the upper confidence bound equation; the higher c is, the more the agent will explore
        tf: t-final, or the number of iterations that the Q-learning goes through (1 iteration = 1 blackjack hand)
        rules: Rules object representing the set of rules to use for the blackjack game (when rules = None, the default constructor for Rules() is used)

    returns results, q_table
        results: a list of length tf containing the bet won/lost at the end of each hand during the training; can be used to create a learning curve
        q_table: a Pandas DataFrame that contains action-state values for the MDP that represents the blackjack game
            columns are actions and rows are states
            the values in the cells are actually tuples: (Q,n)
                Q is the action-state value
                n is the number of times the given action has been chosen in the given state
            we take out the n values from the cells before saving it as a .csv
        
    '''    

    if rules is None: rules = bj.Rules()


    # definte unique states of the game given upcard, player hand, and discard pile
    class State:
        def __init__(self,upcard,player_hand):
            # defining the dealer upcard
            self.upcard = '10' if upcard.num in ['J','K','Q'] else upcard.num
            self.split=False

            # defining player hands; unique cases are pairs, hands with aces, or simply sums
            if len(player_hand.cards) == 2 and player_hand.cards[0].num == player_hand.cards[1].num:
                self.hand = player_hand.cards[0].num + ' ' + player_hand.cards[1].num
                self.split = True
            elif 'A' in [card.num for card in player_hand.cards]: # or we have an ace in the hand
                other_cards = player_hand.get_nums()
                other_cards.remove('A')
                other_cards = [1 if other_cards[n]=='A' else 10 if other_cards[n] in ['J','K','Q'] else int(other_cards[n]) for n in range(len(other_cards))]
                val = sum(other_cards)
                if val > 10: self.hand = str(val+1) # hard ace (not usable) so we count it as one
                else: self.hand = 'A '+str(val) # soft ace (usable) so it gets it own state
                
            else: # if neither of the above are true, the hand is just defined by the sum of the values
                self.hand = str(player_hand.value())

            

        # define our state in terms of attributes        
        def __key(self):
            return (self.upcard,self.hand)

        # create a hash integer based on our key, used for indexing in the q-table
        def __hash__(self):
            return hash(self.__key())

        # test equality of states
        def __eq__(self,other):
            return self.__key() == other.__key()

        # string representation
        def __str__(self):
            return 'upcard: '+self.upcard + ' hand: ' + self.hand
        __repr__ = __str__

    def init_q(): # initialize Q table with actions, but no states
        actions = ['stand','hit','double down','split','surrender']
        return pd.DataFrame(columns=actions)

    # calculate the upper confidence bound for an action-state pair; this takes the Q-value and increases it proportional to the fraction of times it's been chosen
    def UCB(q,n,t):
        if n==0: n=0.0000000001 # since we can't have 0 in the denominator, set an arbitrarily small value so that this value is guaranteed to be the highest
        return q + c*(2*math.log(t)/n)**0.5 # return action-state value adjusted by confidence interval

    # use the blackjack state and q_table to select an action
    def player_move(state,q_table,t,actions):
        if state not in q_table.index: # if we haven't seen this state yet, set each (q,t) pair to (0,0) for that state
            indexes = list(q_table.index) # doing the append removes the index names, so we have to save them
            q_table = q_table.append(pd.Series([(0,0)]*5,index=q_table.columns),ignore_index=True) # append our new state to the q_table with all zeroes
            q_table.index = indexes+[state] # reapply our indexes
            return np.random.choice(actions),q_table # pick a random action from empty columns, since we have no q-values
        
        # we have to use upper confidence bound action selection to chose our action
        options = q_table.loc[state,actions] # get the Q-values for our options
        for i,qn in enumerate(options): # for index and action-state value + number of occurrences tuple
            q,n = qn # q-value and number of times used for each action-state
            options.iloc[i] = UCB(q,n,t) # replace q-values with UCB-adjusted values
        argmax = options.max() # find the largest value
        actions = list(options[options==argmax].index) # get the action(s) where the UCB value is greatest
        i = np.random.randint(len(actions)) # if more than one action, arbitrarily pick, else return the action
        return actions[i],q_table

    # decision logic for testing the Q-table mid train, does not use UCB
    def test_move(state,q_table,actions):
        if state not in q_table.index:
            return np.random.choice(actions)
        
        q_row = q_table.loc[state,actions]
        options = list(q_row[q_row==q_row.max()].index)
        return np.random.choice(options)
                        

    q_table = init_q() # get our state-less q-table
    game = bj.Blackjack(bet=100,rules=rules) # instantiate the blackjack object
    results = [] # append wins and losses here
    tests = [] # results from test runs every 1 mil iterations
    
    
    for t in range(1,tf+1): # play tf number of games
        if t%100000 == 0: print(t)
        if t%1000000 == 0: # every 1 mil iterations, run a test on the q table
            test = bj.Blackjack(bet=100,total_bet=True)
            test_results = []
            test_bet = []
            for n in range(1000000):
                test.start_hand()
                r = test.results()
                while r is None:
                    state = State(test.dealer.hand.cards[0],test.player.hands[test.current_hand])
                    action = test_move(state,q_table,test.available_actions())
                    test.play_action(action)
                    r = test.results()
                if type(r) is float or type(r) is int:
                    r,bet = r,100
                else:
                    r,bet = r
                test_results.append(r)
                test_bet.append(bet)
            returns = (sum(test_results)/sum(test_bet))
            tests.append((t,returns))
            print(returns)
            

                
        as_pairs = [] # keep track of all actions-states chosen in form (action,state)
        game.start_hand()
        result = game.results()
        
        while result is None: # play the hand while it's not over
            state = State(game.dealer.hand.cards[0],game.player.hands[game.current_hand]) # initialize our state with upcard, current player hand, and discard pile
            action,q_table = player_move(state,q_table,t,game.available_actions()) # get action from q-table using UCB
            as_pairs.append((action,state)) # add this transition to our list (we have to wrap up the board in its hash)
            
            game.play_action(action) # play our move
            result = game.results() # get the result (money if the hand is over, otherwise None)

        # now we adjust each of the action-state steps taken according to the result of the game
        results.append(result)
        for a,s in as_pairs: # for each action-state transition from the last game...
            q,n = q_table.loc[s][a] # get the old q-n pair
            q_table.loc[s][a] = ((q*n+result)/(n+1),n+1) # update the q-table with the new average, and increase the n value by 1

    for t,returns in tests:
        print(str(t)+':',returns,'...',end= ' ')
    print()

    # after all games are played, return the list of money made
    return results,q_table


if __name__ == '__main__':
    length = 20000000 # length of learning curve (number of games per learn)
    c=150
    
    # rules of the game that the model will be learning with
    rules = bj.Rules(numdecks=2,double_after_split=True,resplit_aces=False,hit_split_aces=False,late_surrender=False,cut=0.5,bj_payout=3/2,split_hands=4)


    # use these lines for making a .csv from the Q-table of only one learn
    results,q_table = learn_game(c=c,tf=length,rules=rules)
    q_table.applymap(lambda x: x[0]).to_csv('bj_q_table_bs.csv') # we take out the n value from the table, only leaving the Q value, and save it as a .csv
    avgs = [sum(results[n:n+100000])/100000 for n in range(0,20000000-100000,100000)]
    plt.plot(avgs)
    plt.title('Learning Curve for 1 On-policy Learn of 20 mil Hands, Basic Strategy')
    plt.ylabel('Reward from initial bet of 100')

    plt.show()

    
 

    
