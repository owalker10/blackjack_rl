import blackjack_object as bj
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import basicstrategy as bs
import pandas as pd

cvmodel = joblib.load('cvmodel.pkl')

def main(q_table='bj_q_table10.csv',num_iters=100000,kind='q',cvbet=False,nocv=False,rules=bj.Rules(),onehand=None):
    '''
    simulates blackjack games and returns lists of the money won and bet for each hand

    paramters:
        q_table: .csv file containing the trained Q-table
        num_iters: number of hands to play
        kind:
            == 'q': uses specified Q-table to make decisions
            == 'bs': uses basicstrategy.py (basic strategy chart) to make decisions
        cvbet:
            == True: uses the calculated Cerebri Value to place bets
            == False: places a constant bet (arbitrality set to $100)
        nocv:
            == True: ignores the Cerebri Value when creating states (only True when using a basic strategy Q-table aka q_table = 'bj_q_table_bs.csv' and kind='q')
            == False: uses the Cerebri Value when creating states
        rules: Rules object representing the set of blackjack rules to be used
        onehand: used to repeatedly simulate hands of one card combination
            correct implementation is   onehand = [[<player card 1>,<player card 2>],<upcard>] where all the specified cards are strings
            ex: onehand = [['A','2'],'8']    hands will always start with the player having an ace and 2, with the dealer showing an 8 (hole card is allowed to change)

        returns total_won,total_bet
            total_won: list of integers of length <num_iters>, where each element represents the money won for each hand
            total_bet: list of integers of length <num_iters>, where each element represents the money bet for each hand
    '''
    
    global cvmodel
    q_table = pd.read_csv(q_table,index_col=0)
    bet=100
    total_won,total_bet = [],[]

    # consult the Q-table for the best move
    def get_action(upcard,hand,deck,actions):
        state = State(upcard,hand,deck.discard_vals(),nocv=nocv)
        key = str(state)
        q_row = q_table.loc[key][actions]
        moves = list(q_row.index[q_row==q_row.max()])
        return np.random.choice(moves)
    # consult the basis strategy table for the best move
    def get_bs(upcard,hand,num_hands,rules):
        upcard = upcard.num
        return bs.get_move(hand,upcard,num_hands,rules=rules)

    game = bj.Blackjack(bet=bet,total_bet=True,cvbet=cvbet,rules=rules,onehand=onehand)
    start_money = game.player.money
    for n in range(num_iters):
        if onehand is None: game.start_hand()
        else: game.reset_hand()

        result = game.results()

        while result is None:

            if kind == 'bs': action = get_bs(game.dealer.hand.cards[0],game.player.hands[game.current_hand],len(game.player.hands),rules)
            elif kind == 'q': action = get_action(game.dealer.hand.cards[0],game.player.hands[game.current_hand],game.deck,game.available_actions())
            game.play_action(action)
            result = game.results()

        if type(result) is int or type(result) is float:
            result = result,bet

        won = result[0]
        amt_bet = result[1]



        total_won.append(won)
        total_bet.append(amt_bet)



    return total_won,total_bet

def find_diffs(q_table='bj_q_table9.csv',num_iters=10000):
    '''
    simulates blackjack games and keeps track of when basic strategy and trained Q-learning do and don't agree

    paramters:
        q_table: .csv file containing the trained Q-table
        num_iters: number of hands to play

    returns diff_table, frac_diff:
        diff_table: Pandas DataFrame where the rows and columns are the same as the Q-table (states and actions); the cells themselves are one of the following:
            blank: a space character, default empty cell
            'BS': the basic strategy table chose this cell's column as the correct action for the state as given by the cell's row, but the Q-table didn't
            'Q': the Q-table chose this cell's column as the correct action for the state as given by the cell's row, but the basic strategy table didn't
            'BS/Q': both the Q-table and basic strategy table agree that the given action is optimal for the given state
        frac_diff: fraction of times that basic strategy and the Q-table disagreed on the correct action to take
    '''
    
    global cvmodel
    q_table = pd.read_csv(q_table,index_col=0)
    bet=100
    total_won,total_bet = [],[]


    def get_action(upcard,hand,deck,actions):
        state = State(upcard,hand,deck.discard_vals())
        key = str(state)
        q_row = q_table.loc[key][actions]
        moves = list(q_row.index[q_row==q_row.max()])
        return np.random.choice(moves)

    def get_bs(upcard,hand,num_hands):
        upcard = upcard.num
        return bs.get_move(hand,upcard,num_hands)

    game = bj.Blackjack(bet=bet,total_bet=True)
    start_money = game.player.money
    count=0
    total=0
    diff_table = q_table.applymap(lambda x: ' ')
    for n in range(num_iters):
        game.start_hand()

        result = game.results()

        while result is None:

            bs_action = get_bs(game.dealer.hand.cards[0],game.player.hands[game.current_hand],len(game.player.hands))
            q_action = get_action(game.dealer.hand.cards[0],game.player.hands[game.current_hand],game.deck,game.available_actions())
            state = State(game.dealer.hand.cards[0],game.player.hands[game.current_hand],game.deck.discard_vals())
            if bs_action != q_action:
                diff_table.loc[str(state),bs_action] = 'BS'
                diff_table.loc[str(state),q_action] = 'Q'
                count+=1
            else:
                diff_table.loc[str(state),bs_action] = 'BS/Q'
            if bs_action in ['surrender','double down','split'] and len(game.player.hands[game.current_hand].cards) >2:
                print('cheater',bs_action,game.player.hands[game.current_hand].cards)
            if bs_action == 'double down' and '18' in str(state):
                print(q_action,state,game.player.hands[game.current_hand].cards)
                
                
            total+=1
            game.play_action(q_action)
            result = game.results()

        if type(result) is int or type(result) is float:
            result = result,bet

        won = result[0]
        amt_bet = result[1]

        total_won.append(won)
        total_bet.append(amt_bet)



    return diff_table,(count/total)

# given the discard pile, calculate the raw Cerebri Value   
def calc_cv(discard):
    global cvmodel
    cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    vector = [discard.count(card) for card in cards]
    cv = cvmodel.predict_proba([vector])[0][1]
    return cv

# given a dataset of discard piles and a set of quantile boundaries in terms of percent, return the Cerebri Values at the quantiles boundaries
def literal_quantiles(data='discard_data_1000000.csv',quantiles=[10,33,50,66,90]):
    discards = pd.read_csv(data)[['A','2','3','4','5','6','7','8','9','10','J','Q','K']]
    cvs = cvmodel.predict_proba(discards).T[1]
    return list(np.percentile(cvs,quantiles))

quantiles = literal_quantiles(quantiles=[33,66])

# change the quantiles
def change_quantiles(qs):
    global quantiles
    quantiles = literal_quantiles(quantiles=qs)



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


if __name__ == '__main__':
    '''
    change_quantiles([33,66])
    won,bet = main(q_table='bj_q_table1-2.csv',num_iters=3000000,kind='q',nocv=False,rules=bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,
                                                                                                             resplit_aces=False,cut=0.5,late_surrender=False))

    print('Jokers Wild Casino, CV Model:')
    print(str(round(sum(won)/sum(bet)*100,4))+'%')
    '''

