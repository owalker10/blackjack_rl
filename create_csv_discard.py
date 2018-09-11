import blackjack_object as bj
import pandas as pd
import numpy as np

'''
Simulates hands of blackjack and takes the beginning state of the discard pile and the final result of the hand as data
Then converts the discard pile data into one-hot encoded values by individual card
This data gets saved as a .csv
'''

cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']


# return the number of occurences of card in cards
def freq(discards):
    vector = []
    for card in cards:
        vector.append(discards.count(card))
    return pd.Series(vector,index=cards)

# change the label from amount of money made to 1 for net gain and 0 for net loss
def label_transform(label):
    if label>0:
        return 1
    return 0


def simulate(q_table='bj_q_table_bs.csv',num_iters=100000,kind='q',rules=bj.Rules(),bet=100):
    q_table = pd.read_csv(q_table,index_col=0)
    
    def get_action(upcard,hand,deck,actions):
        state = State(upcard,hand,deck.discard_vals(),nocv=True)
        key = str(state)
        q_row = q_table.loc[key][actions]
        moves = list(q_row.index[q_row==q_row.max()])
        return np.random.choice(moves)

    def get_bs(upcard,hand,num_hands):
        upcard = upcard.num
        return bs.get_move(hand,upcard,num_hands)

    discard_piles = []
    results = []

    game = bj.Blackjack(bet=bet,total_bet=False,rules=rules)
    start_money = game.player.money
    for n in range(num_iters):
        game.start_hand()
        discard_piles.append(game.deck.discard_vals())

        result = game.results()

        while result is None:

            if kind == 'bs': action = get_bs(game.dealer.hand.cards[0],game.player.hands[game.current_hand],len(game.player.hands))
            elif kind == 'q': action = get_action(game.dealer.hand.cards[0],game.player.hands[game.current_hand],game.deck,game.available_actions())
            #print(action,len(game.player.hands),'player:',[card.num for card in game.player.hands[game.current_hand]],'| dealer:',[card.num for card in game.dealer.hand])
            game.play_action(action)
            result = game.results()
            
        results.append(result)

    data = pd.DataFrame([discard_piles,results]).T ; data.columns = ['discards','result']
    discard_piles,results = None,None
    data.loc[:,'result'] = data.loc[:,'result'].apply(pd.to_numeric,downcast='integer') # downsize the storage type to prevent memory errors
    return data

class State:
    '''
    represents a state from the state set of the environment

    init parameters:
        upcard: Card object that represents the dealer upcard
        player_hand: Hand object that represents the player hand
        discard: list of strings that represent the values of the cards in the discard pile
        nocv: whether the object should ignore the Cerebri Value of the discard pile when generating the state (when nocv == True, you're effectively using basic strategy)
    '''
    def __init__(self,upcard,player_hand,discard,nocv=False):
        # defining the dealer upcard
        self.upcard = '10' if upcard.num in ['J','K','Q'] else upcard.num
        self.is_split = False

        # defining player hands; unique cases are pairs, hands with aces, or simply sums
        if len(player_hand.cards) == 2 and player_hand.cards[0].num == player_hand.cards[1].num:
            self.hand = player_hand.cards[0].num + ' ' + player_hand.cards[1].num
            self.is_split = True
        elif 'A' in [card.num for card in player_hand.cards]: # or we have an ace in the hand
            other_cards = player_hand.get_nums()
            other_cards.remove('A')
            other_cards = [1 if other_cards[n]=='A' else 10 if other_cards[n] in ['J','K','Q'] else int(other_cards[n]) for n in range(len(other_cards))]
            val = sum(other_cards)
            if val > 10: self.hand = str(val+1) # hard ace (not usable) so we count it as one
            else: self.hand = 'A '+str(val) # soft ace (usable) so it gets it own state
        else: # if neither of the above are true, the hand is just defined by the sum of the values
            self.hand = str(player_hand.value())

        
        # defining the CV in terms of quantiles
        if not nocv: cv = calc_cv(discard)
        # self.cv_bin will be [0,n-1] for having n quantiles
        if not nocv: self.cv_bin = sum([1 if cv >= quantiles[n] else 0 for n in range(len(quantiles))])
        self.nocv = nocv

    # define our state in terms of attributes        
    def __key(self):
        return (self.upcard,self.hand,self.cv_bin) if not self.nocv else (self.upcard,self.hand)

    # create a hash integer based on our key, used for indexing in the q-table
    def __hash__(self):
        return hash(self.__key())

    # test equality of states
    def __eq__(self,other):
        return self.__key() == other.__key()

    def __str__(self):
        return ('upcard: '+self.upcard + ' hand: ' + self.hand + ' cv bin: ' + str(self.cv_bin)) if not self.nocv else ('upcard: '+self.upcard + ' hand: ' + self.hand)
    __repr__ = __str__


# defines the set of rules this simulation will play by
rules = bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False)

# simulate the data; will contain discard piles and money won/lost per hand
data = simulate(num_iters=1000000,rules=rules)
#print(data.head())

# transform the data

# change the discard data from a one-column list to a 13 column one-hot vector
data.loc[:,cards] = data['discards'].apply(freq) # make columns for each card type and fill them with the number of occurences
data.loc[:,cards] = data.loc[:,cards].applymap(pd.to_numeric,downcast='unsigned') # downsize the storage type to prevent memory errors

data.drop(axis=1,columns='discards',inplace=True)

#data = data.loc[:,data.columns[data.columns!='discards']] # don't need this column anymore (using .drop() was throwing a memory error)
# change the result column from money to 0 for loss and 1 for win (we drop ties)
data = data[data['result']!=0] # drop all rows where net gain is 0 (we want a binary classifier for wins/losses)
data['result'].astype('category',copy=False)
data['result'] = data['result'].apply(label_transform) # change the label from amount of money made to 1 for net gain and 0 for net loss


#print(data.head())

#data.to_csv('discard_data_1000000.csv') # UNCOMMENT ME TO SAVE THE DATA AS A CSV
