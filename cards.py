import random
from random import Random

'''
Playing card module for Blackjack
'''

# possible numbers/faces for cards
NUMS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
# possible suits for cards
SUITS = ['spades','hearts','clubs','diamonds']
# maps string number\face value of card to numerical value
TO_VAL = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':10,'Q':10,'K':10,'A':11}
# starting money for game
STARTING_MONEY = 500

def winner(player,dealer,num_hand):
    player_hand = player.hands[num_hand]
    dealer_hand = dealer.hand
    if player_hand.busted(): # if player is busted
        return dealer # dealer wins
    elif dealer_hand.busted(): # if dealer is busted (player is not)
        return player # player wins
    elif player_hand.value() > dealer_hand.value(): # neither are busted, player has higher hand
        return player # player wins
    elif player_hand.value() < dealer_hand.value(): # neither are busted, dealer has a higher hand
        return dealer # dealer wins
    return None # equal hands, nobody wins


class Card(object):
    '''
    card object, represents one playing card
    '''
    # num is a string
    def __init__(self,num,suit):
        self.num = num
        self.suit = suit

    def __eq__(self,other):
        return self.num == other.num and self.suit == other.suit

    def __str__(self):
        return '{} of {}'.format(self.num,self.suit)
    __repr__ = __str__


class Deck(object):
    '''
    deck object, represents a deck with:
        deck: list of cards in deck to be drawn from, starts with 52 cards unless otherwise specified
        discard: list representing discard pile, gets shuffled into deck when deck runs out
        in_play: list of cards currently in play, gets put into discard on end_hand() call
        cards_played: number of total cards drawn from deck
        self.rand: random object used if seed is specified
    '''
    def __init__(self,seed=None,cards=None,numdecks=1,cut=None):
        self.deck = [];self.discard = [];self.in_play = []
        self.rand=None
        if seed is not None:
            self.rand = Random(seed)        
        if cards is not None: # if we want to make a very specific deck with given cards
            deck,discard,in_play = cards
            suits = [0 for n in range(len(NUMS))]
            for card in deck:
                self.deck.append(Card(card,SUITS[suits[NUMS.index(card)]]))
                suits[NUMS.index(card)]+=1
            for card in discard:
                self.discard.append(Card(card,SUITS[suits[NUMS.index(card)]]))
                suits[NUMS.index(card)]+=1
            for card in in_play:
                self.in_play.append(Card(card,SUITS[suits[NUMS.index(card)]]))
                suits[NUMS.index(card)]+=1
            self.shuffle()
                
        else:  
            for n in NUMS:
                for s in SUITS:
                    self.deck.append(Card(n,s))
            self.deck = self.deck * numdecks
      
            self.shuffle()
            self.discard = []
            self.in_play = []
        self.cards_played = 0
        self.cut=cut

            
    def shuffle(self):
        if self.rand is not None:
            self.rand.shuffle(self.deck)
        else:
            random.shuffle(self.deck)
    def get_card(self):
        self.cards_played += 1
        if self.deck == [] or self.deck is None: # put discard pile back into deck and shuffle
            self.deck.extend(self.discard)
            self.discard.clear()
            self.shuffle()
        # if there's a cut card, check to see if we've reached it. if so, reshuffle the deck
        if self.cut is not None:
            if len(self.deck)/52 <= self.cut:
                self.deck.extend(self.discard)
                self.discard.clear()
                self.shuffle()
        card = self.deck.pop()
        self.in_play.append(card) # the card is currently in play
        return card
    def end_hand(self): # called when hand is over, before new hands are dealt
        self.discard.extend(self.in_play)
        self.in_play.clear()
    def get_hand(self):
        return Hand(self)
    def discard_vals(self):
        return [self.discard[n].num for n in range(len(self.discard))] # give a list of cards in string form

class Hand(object):
    '''
    hand object, represents blackjack hand of 2+ cards
    '''
    def __init__(self,deck,split_card = None,nodraw=False):
        self.deck = deck
        self.ace = False
        if nodraw:
            self.cards = []
        else:
            # if the hand is being created from a split, make the first card the card from the original hand (else draw it from deck)
            self.cards = [self.deck.get_card()] if split_card is None else [split_card]
            self.cards.append(self.deck.get_card())
        
    def hit(self): # hit: draw a card from deck
        self.cards.append(self.deck.get_card())
        if self.cards[-1].num == 'A': self.ace = True
    def value(self): # returns hand value
        self.val = 0
        for card in self.cards:
            self.val+= TO_VAL[card.num]
        for card in self.cards: # if there are aces, minimize the hand until no longer busted
            if card.num == 'A' and self.val > 21:
                self.val -=10
        return self.val
    def card(self,n): # return nth card
        return self.cards[n]
    def busted(self):
        return self.value() > 21
    def blackjack(self):
        return ((len(self.cards) == 2) and (self.value() == 21))
    def get_nums(self):
        return [self.cards[n].num for n in range(len(self.cards))]
    def __str__(self):
        return str(self.cards)
    def __repr__(self):
        return repr(self.cards)
    # iter and next to allow Hand() objects in for each loops
    def __iter__(self):
        return self.cards.__iter__()
    def __next__(self):
        return self.cards.__next__()



class Player(object):
    '''
    player object, represents a player with:
        name: player name
        money: total money player can bet with
        hand: first hand per round
        hands: list of hands (will be more than one if player splits)
        deck: deck that player draws from
    '''
    def __init__(self,name,deck):
        self.name = name
        self.money = STARTING_MONEY
        self.hand = None
        self.hands = []
        self.deck = deck
        self.total_bet = 0
    def new_hand(self):
        self.hand = Hand(self.deck)
        self.hands = [self.hand]
    def reset_hand(self,cards): # used only when you want the player hand to be a specific combination of cards
        self.hand = Hand(self.deck,nodraw=True)
        self.hand.cards.extend(cards)
        self.hands = [self.hand]
    def bet(self,amount):
        self.money-= amount
        self.total_bet+=amount
        return amount
    def hit(self,hand=None):
        if hand is None:
            hand = self.hand
        hand.hit()
    def split(self,hand): # make a new hand, using the second card of the first hand, and finish both with deck draws
        self.hands.append(Hand(self.deck,split_card = hand.cards[1]))
        hand.cards[1] = self.deck.get_card()
    def is_split(self):
        return len(self.hands) > 1
    def num_hands(self):
        return len(self.hands)
    def blackjack(self): # blackjack only if the hand is not split, has two cards, and equals 21
        return ((len(self.hand.cards) == 2) and (self.hand.value() == 21) and not self.is_split() and len(self.hands)==1)
    def card(self,n):
        return self.hand.cards[n]
    def __str__(self):
        return '{} | {} | {}'.format(self.name,self.hand,self.money)
