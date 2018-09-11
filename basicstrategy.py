import pandas as pd
import cards as cd
import blackjack_object as bj

'''
this script contains methods for parsing decisions from a basic strategy chart, and does nothing when run
'''

chart = pd.read_csv('basic_strategy.csv',index_col=0)

TO_VAL = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':10,'Q':10,'K':10,'A':11}


def get_value(hand):
    '''
    calculates and returns the value of the hand; if aces are present, returns highest possible value below 22

    parameters:
        hand: Hand object to calculate value from
    returns: value of hand
    '''
    val = 0
    for card in hand:
        val+= TO_VAL[card.num]
    for card in hand: # if there are aces, minimize the hand until no longer busted
        if card == 'A' and val > 21:
            val -=10
    return val


def get_bs_move(player_hand,dealer_upcard,numberhands,rules=bj.Rules()):
    '''
    given the cards on the table and the rules of the game, return the best move according to basic strategy

    parameters:
        player_hand: Hand object representing the player hand
        dealer_upcard: string representing the dealer's upcard
        numberhands: the number of hands the player currently has
        rules: Rules object containing the set of rules for the current game
        
    returns: a string representing the best move to make
    '''
    global chart
    cards = []
    for card in player_hand.cards:
        cards.append(card.num)
    cards = ['10' if card in ['J','K','Q'] else card for card in cards]
    dealer_upcard = '10' if dealer_upcard in ['J','K','Q'] else dealer_upcard
        
    if len(cards)==2: # this will deal with matching pairs of cards or aces plus one other card
        cards.sort(); cards.reverse();
        pair = cards[0]+cards[1]
        # makes sure the pairing is in the chart, and the computer isn't trying to split into too many hand
        if pair in chart.index and not (chart.loc[pair,dealer_upcard] == 'split' and numberhands >= rules.split_hands):
            move = chart.loc[pair,dealer_upcard]
            if pair == 'AA' and numberhands>1 and not rules.resplit_aces:
                if rules.hit_split_aces: return 'hit'
                else: return 'stand'
            if cards[0] == 'A' and numberhands>1 and not rules.hit_split_aces: return 'stand'
            if '/' in move: return move.split('/')[0] # return the first option in the table (second is if you have more than two cards)
            else: return move
        elif pair == 'AA': return 'hit'
    if 'A' in cards: # deals with ace plus at least two other cards
        new_cards = cards[:]
        new_cards.remove('A')
        vals = [1 if new_cards[n]=='A' else int(new_cards[n]) for n in range(len(new_cards))]
        value = 'A'+str(sum(vals)) if sum(vals) < 11 else sum(vals)+1
        move = chart.loc[str(value),dealer_upcard]
        if '/' in move: return move.split('/')[1] # return the second option since we have more than two cards
        else: return move
    else: value = player_hand.value() # if there are no pairs or aces in hand, just take the sum
    move = chart.loc[str(value),dealer_upcard]

    # check to see if there are two options, and if so which one we should play, otherwise just return the one move
    if '/' in move:
        if len(cards) == 2: return move.split('/')[0]
        else: return move.split('/')[1]
    else: return move



# make move based upon logic passed (no other kind of decision making has been supported besides BS)
def get_move(player_hand,dealer_upcard,numberhands,kind='bs',rules=bj.Rules()):
    if kind is 'bs':
        return get_bs_move(player_hand,dealer_upcard,numberhands,rules=rules)
    #elif kind is ...
