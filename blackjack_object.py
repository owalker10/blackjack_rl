import cards as cd
import pandas as pd
import copy
from sklearn.externals import joblib
import numpy as np

cd.STARTING_MONEY = 100000

'''
This script contains several objects that represent items in blackjack. Running this script does nothing.
- Blackjack: represents a full game of blackjack including rules, players, decks, etc.
- State: represents a state in the state set of the reinforcement learning environment (input cards and the class creates a key() based on the determined state)
- Rules: represents a collection of variable rules in blackjack
'''

'''
Class that represents a game of Blackjack

'''
class Blackjack(object):
    '''
    init parameters:
        bet: starting bet for each hand, if betting isn't based on the Cerebri Value
        total_bet: if True, upon completion of a hand, the results() method will return <money_won, money_bet>. If False, money_won is returned only.
        cvbet: if True, object will use the calculated Cerebri Value of the current discard pile to place adjusted bets (similar to card counting)
        rules: Rules object representing the set of rules of the game to be instantiated (see Rules class below for specifics)
        onehand: used to repeatedly simulate hands of one card combination
            correct implementation is   onehand = [[<player card 1>,<player card 2>],<upcard>] where all the specified cards are strings
            ex: onehand = [['A','2'],'8']    hands will always start with the player having an ace and 2, with the dealer showing an 8 (hole card is allowed to change)
    '''
    def __init__(self,bet=10,total_bet=False,cvbet=False,rules=None,onehand=None):
        if rules is None: rules = Rules()
        self.rules = rules
        self.deck = cd.Deck(numdecks=self.rules.numdecks,cut=self.rules.cut) # deck for the game
        self.player = cd.Player('learner',self.deck) # the player; in this case, the learning agent
        self.dealer = cd.Player('dealer',self.deck) # the dealer
        self.initial_bet = bet
        self.bet = [self.initial_bet] # the list of bet for each hand in the player's hands
        self.stand=[False] # whether or not the hand has ended for each hand in the player's hands
        self.surrender=[False] # whether or not the player surrendered for each hand in the player's hand
        self.current_hand=0 # the current hand being played
        self.end = False # whether or not the round has ended
        self.total_bet = total_bet
        self.cvmodel = joblib.load('cvmodel.pkl')
        self.cvbet = cvbet
        #print(rules)

        if onehand is not None:
            hand = onehand[0]
            upcard = onehand[1]
            for card in hand+[upcard]: self.deck.deck.remove(card)
            self.hand = hand
            self.upcard = upcard
    
    def cv_bet(self,discards):
        '''
        given a discard pile, use cvmodel.pkl's probability function to calculate a raw Cerebri Value and translate it to a betting amount

        parameters:
            discards: a list of strings, where each element represents the value of each card in the discard pile

        returns: integer value representing the dollar amount to be bet
        '''
        cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        vector = [discards.count(card) for card in cards] # each card becomes a 13 length one-hot encoded vector, and then all these vectors are summed
        cv_raw = self.cvmodel.predict_proba([vector])[0][1] # use the positive probability of the classifier model for the raw CV
        cv = (cv_raw-0.50)/(0.55-0.50) # all raw CV's under 0.50 will map to a negative value (0.45 -> -1, 0.50 -> 0, 0.55 -> 1, and so forth)
        cv = cv*496+4 # our bet will fall within $4 and $500 for the most part, but CV's over 0.55 wil exceed $500 (that's fine)
        if cv < 4: # if our probability of winning is less than 50%, bet the minimum (we can't be negative)
            cv = 4

        return int(round(cv)) # we bet in integers

    #start a new hand
    def start_hand(self):
        self.deck.end_hand()
        self.player.new_hand()
        self.dealer.new_hand()

        self.starting_money = self.player.money
        
        if not self.cvbet:
            self.bet = [self.initial_bet]
        else:
            self.bet = [self.cv_bet(self.deck.discard_vals())]
        self.player.bet(self.bet[0])
        self.stand = [False]
        self.surrender = [False]
        self.current_hand = 0

        if self.player.blackjack() or self.dealer.blackjack():
            self.end = True
        else: self.end = False
        
    # this method is ONLY used when you're doing repeated simulations of one specific hand (aka the init parameter onehand is not None)
    def reset_hand(self):
        self.deck.end_hand()
        self.player.reset_hand(self.hand)
        self.dealer.hand = cd.Hand(self.deck,nodraw=True); self.dealer.hand.cards = [self.upcard,self.deck.get_card()]

        self.starting_money = self.player.money
        
        if not self.cvbet:
            self.bet = [self.initial_bet]
        else:
            self.bet = [self.cv_bet(self.deck.discard_vals())]
        self.player.bet(self.bet[0])
        self.stand = [False]
        self.surrender = [False]
        self.current_hand = 0

        if self.player.blackjack() or self.dealer.blackjack():
            self.end = True
        else: self.end = False
        


    # create a split from the nth hand
    def make_split(self,n):
        self.bet.append(self.player.bet(self.bet[n])) # put an additional bet on the new hand
        
        self.stand.append(False if not self.player.hands[n].value() == 21 else True) # if you have 21, you don't want to play the hand
        self.surrender.append(False) # we haven't surrendered yet for this hand

    # based on the current state of the game and the rules of the game, determine which moves are allowed
    def available_actions(self):
                    
        actions = ['stand','hit','split','double down','surrender']
        if len(self.player.hands[self.current_hand].cards) > 2:
            actions = ['stand','hit']
        # if surrendering isn't allowed, take it out of the actions
        if 'surrender' in actions and self.rules.late_surrender == False: actions.remove('surrender')
        card1,card2 = self.player.hands[self.current_hand].cards[0].num, self.player.hands[self.current_hand].cards[1].num
        # you can split when you have less than four hands, and when your two cards match (a 10 and K are considered matching)
        if 'split' in actions and ((card1!=card2 and (card1 not in ['10','J','Q','K'] or card2 not in ['10','J','Q','K'])) or len(self.player.hands) >= self.rules.split_hands):
            actions.remove('split')
        # check for the rule that prevents splitting aces that were already split
        if 'split' in actions and not self.rules.resplit_aces and card1 == 'A' and len(self.player.hands)>1:
            actions.remove('split')
        # if doubling down after splitting isn't allowed and you've split, take it out of actions
        if 'double down' in actions and not self.rules.double_after_split and len(self.player.hands) > 1: actions.remove('double down')
        # check for the rule that prevents hitting aces after splitting them (also includes doubling down)
        if 'hit' in actions and not self.rules.hit_split_aces and len(self.player.hands)>1 and card1 == 'A':
            actions.remove('hit')
        if 'double down' in actions and not self.rules.hit_split_aces and len(self.player.hands)>1 and card1 == 'A':
            actions.remove('double down')        
            
        return actions


    
    # given an action in string form, play the move, and end the game if appropriate
    def play_action(self,action):
        if len(self.player.hands)>1 and self.player.hands[self.current_hand].cards[0].num == 'A' and action=='hit': print('BAD',self.rules.hit_split_aces,self.player.hands)
        #print(self.player.hands[self.current_hand],self.dealer.hand.cards[0],action)
        if action == 'stand': # player stands: their hand is done for the round
            self.stand[self.current_hand] = True
        elif action == 'hit': # player hits: their hand gains a card from the deck
            self.player.hit(hand=self.player.hands[self.current_hand])
            if self.player.hands[self.current_hand].busted(): # if the player busted on this hit, their hand is done
                self.stand[self.current_hand] = True
        elif action == 'double down': # player doubles down: they double their bet on the hand, hit, and stand
            self.bet[self.current_hand]+=self.player.bet(self.bet[self.current_hand]) # double the bet
            self.player.hit(hand=self.player.hands[self.current_hand]) # give the player a card
            self.stand[self.current_hand] = True # they're now standing
        elif action == 'split': # player splits two equal cards: they split current hand into 2 new ones
            self.player.split(self.player.hands[self.current_hand])
            self.make_split(self.current_hand)
        elif action == 'surrender': # player surrenders on first turn: gaining back half their bet
            self.surrender[self.current_hand] = True
            self.stand[self.current_hand] = True

        if self.stand[self.current_hand]: # if this hand is over, either move to the next one, or if this was the last hand, end the game
            if self.current_hand == len(self.stand)-1: self.end = True
            else: self.current_hand+=1

        
    # returns the results of the game or None if it isn't over yet
    def results(self):
        if not self.end:
            return None
                
        if self.player.blackjack(): # if player had a blackjack at the start
            if self.dealer.blackjack(): # if the dealer also had a blackjack at the start
                self.player.money+= self.bet[0]
                return (0,sum(self.bet)) if self.total_bet else 0 # nobody wins
            else: # but if the dealer didn't have a blackjack
                self.player.money+=int((1.0+self.rules.bj_payout)*self.bet[0])
                return (int(self.bet[0]*self.rules.bj_payout),sum(self.bet)) if self.total_bet else int(self.bet[0]*self.rules.bj_payout)  # player wins (extra return on blackjacks)
            
        elif self.dealer.blackjack(): # but if the dealer had a blackjack and the player didn't
            return (-self.bet[0],sum(self.bet)) if self.total_bet else -self.bet[0] # dealer wins

        # if no blackjacks, the dealer plays and we go through each hand and tally the winnings

        # dealer must hit until he has 17
        while (self.dealer.hand.value() < 17):
            self.dealer.hit()
        # dealer hits on soft 17 (ace in the hand with value of 11)
        if ('A' in [card.num for card in self.dealer.hand]) and (self.dealer.hand.value() == 17) and self.rules.h17:
            soft = True
            cards = [card.num for card in self.dealer.hand]
            cards.remove('A')
            for card in cards:
                if card in ['J','Q','K']: soft=False
            if soft: soft = sum([(int(card) if card != 'A' else 1) for card in cards]) == 6
            
            if soft:
                self.dealer.hit()

        #print([card.num for card in self.dealer.hand],self.dealer.hand.value())
        
        net = 0
        for n,player_hand in enumerate(self.player.hands):

            if self.surrender[n]: # if the player surrendered, they get half their bet back
                self.player.money+=int(self.bet[n]/2)
                net-= int(self.bet[n]/2)
            else:
                dealer_hand = self.dealer.hand
                
                if player_hand.busted(): # if player is busted
                    net-=self.bet[n] # dealer wins EVEN IF THE DEALER BUSTED
                elif dealer_hand.busted(): # if dealer is busted (player is not)
                    self.player.money+=2*self.bet[n]
                    net+=self.bet[n] # player wins
                elif player_hand.value() > dealer_hand.value(): # neither are busted, player has higher hand
                    self.player.money+=2*self.bet[n]
                    net+=self.bet[n] # player wins
                elif player_hand.value() < dealer_hand.value(): # neither are busted, dealer has a higher hand
                    net-=self.bet[n] # dealer wins
                else:
                    self.player.money+=self.bet[n]
                    net+=0 # equal hands, nobody wins
                    
                    
        return (net,sum(self.bet)) if self.total_bet else net
            



                         
cvmodel = joblib.load('cvmodel.pkl')

# given a list of strings representing the values of the cards in the discard pile, return the raw Cerebri Value (probability)
def calc_cv(discard):
    global cvmodel
    cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    vector = [discard.count(cards[n]) for n in range(len(cards))]
    cv = cvmodel.predict_proba(np.array(vector).reshape(1, -1))[0][1]
    return cv

# represents a state from the state set of the environment (some combinations of cards, while different, may generate the same state; ex: player hand 6 & 7 is identical to 10 & 3)
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

# description of each rule is commented after the assignment in the __init__ function
class Rules(object):
    def __init__(self,numdecks=1,double_after_split=True,hit_split_aces=True,resplit_aces=True,split_hands=4,late_surrender=True,bj_payout=3/2,cut=None,h17=True):
        self.numdecks=numdecks # number of decks to be used
        self.double_after_split=double_after_split # whether the player can double down after splitting their hand
        self.hit_split_aces=hit_split_aces # whether the player can take a third card on hands made from splitting aces
        self.resplit_aces=resplit_aces # whether the player can split a pair of aces that were created by splitting aces
        self.split_hands=split_hands # maximum number of hands a player can have
        self.late_surrender=late_surrender # whether the player can surrender at the beginning of the hand, losing half their bet
        self.bj_payout=bj_payout # multiplier for payouts on player blackjacks (if bj_payout is 3/2, an initial bet of $10 will net you $15 upon getting a blackjack)
        self.cut=cut # where the cut card is in the deck, or None if no cut card (if cut == 0.5, the dealer will reshuffle the discard pile back in when the unplayed pile reaches half a deck
        self.h17 = h17 # whether the dealer hits on a soft 17 (ace plus total sum of 6)

    def __str__(self):
        return ('numdecks = '+str(self.numdecks)+' | double_after_split = '+str(self.double_after_split)+' | hit_split_aces = '+str(self.hit_split_aces)+' | resplit_aces = '+str(self.resplit_aces)+
                ' | split_hands = '+str(self.split_hands)+' | late_surrender = '+str(self.late_surrender)+' | bj_payout = '+str(self.bj_payout)+' | cut = '+str(self.cut)+
                ' | h17 = '+str(self.h17))
        
    

    
