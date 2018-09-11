from matplotlib import pyplot as plt
import numpy as np
import bs_sim_cv_vs_count as sim
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.externals import joblib
import blackjack_object as bj
import pandas as pd

'''
visually demonstrate the Cerebri Value and traditional running count over time with a series of methods
'''

def simulate(q_table='bj_q_table_bs.csv',num_iters=30,rules=bj.Rules(),bet=100):
    '''
    simulate blackjack games and net winnings

    parameters:
        q_table: .csv file of the Q-table to be used
        num_iters: number of hands to be simulated
        rules: Rules object representing the rules of the game
        bet: initial bet for every hand

    returns: Pandas DataFrame where every row contains a Cerebri Value and HiLo running count
    '''
    
    q_table = pd.read_csv(q_table,index_col=0)
    cvmodel = joblib.load('cvmodel.pkl')

    # take a list of strings representing the discard pile an turn it into a 13 length vector where each element is the count of a card value
    def vectorize(discard_vals):
        cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        return [discard_vals.count(card) for card in cards]

    # given a lsit of strings representing the discard pile, calculate the raw CV using the CV model's probability function
    def calc_cv(discard_vals):
        vector = vectorize(discard_vals)
        return cvmodel.predict_proba([vector])[0][0]

    # calculate the HiLo running count from a discard pile
    def calc_count(discard_vals):
        return sum([1 if card in ['A','K','Q','J','10'] else -1 if card in ['2','3','4','5','6'] else 0 for card in discard_vals])
        
    def get_action(upcard,hand,deck,actions):
        state = bj.State(upcard,hand,deck.discard_vals(),nocv=True)
        key = str(state)
        q_row = q_table.loc[key][actions]
        moves = list(q_row.index[q_row==q_row.max()])
        return np.random.choice(moves)

    cvs = []
    counts = []
                   
    game = bj.Blackjack(bet=bet,total_bet=False,rules=rules)
    start_money = game.player.money
    for n in range(num_iters):
        game.start_hand()
           

        result = game.results()
        

        while result is None:

            action = get_action(game.dealer.hand.cards[0],game.player.hands[game.current_hand],game.deck,game.available_actions())
            #print(action,len(game.player.hands),'player:',[card.num for card in game.player.hands[game.current_hand]],'| dealer:',[card.num for card in game.dealer.hand])
            game.play_action(action)

            discard = game.deck.discard_vals()
            cvs.append(calc_cv(discard))
            counts.append(calc_count(discard))          

            result = game.results()
            

    data = pd.DataFrame([cvs,counts]).T ; data.columns = ['cv','count']
    return data
    
# shows graph of CV vs. running count over time (hands) using the normally trained CV model
def main():
    hands = 30
    decks = 3
    #data = sim.main(10,decks,count='decks') # generate simulated data
    data = simulate(num_iters=hands,rules=bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False))
    cvs = data['cv']
    counts = data['count']
    print(cvs.max(),cvs.min())

    scaler = MinMaxScaler()

    # normalize both datasets with min max
    cvs = scaler.fit_transform(cvs.values.reshape(-1,1))
    counts = scaler.fit_transform(counts.values.reshape(-1,1))

    # plot data
    ax = plt.gca()
    #ax.set_title(label='CV vs HiLo Card Count Through '+str(decks)+' Decks, Normalized')
    ax.set_title(label='CV vs HiLo Card Count Through '+str(hands)+' Hands, Normalized')
    ax.plot(cvs,label='Cerebri Value');ax.plot(counts,label='Running Count')
    ax.legend()
    plt.show()

# shows graph of CV vs. running count over time (hands) using partially trained models of partial fractions ns
def run_partials(ns):
    # create subplot configuration of appropriate size
    fig,axes = plt.subplots(nrows = math.ceil(len(ns)/3),ncols=3)
    decks = 1
    models = []
    scaler = MinMaxScaler()

    # simulate data with given partial models
    for n in ns:            
        models.append('cvmodel_partial'+str(int(n*100))+'.pkl')
    data = sim.main(10,decks,count='decks',model=models)
    counts = data['count']
    counts = scaler.fit_transform(counts.values.reshape(-1,1))

    # plot data from each partial model
    for i,n in enumerate(ns):
        cvs = data[n]
        cvs = scaler.fit_transform(cvs.values.reshape(-1,1))

        ax = axes[i//2][i%2]
        ax.set_title(label='CV Model Trained and Tested with '+str(int(n*100))+'% of data')
        ax.plot(cvs,label='Cerebri Value');ax.plot(counts,label='Running Count')
        ax.legend()
    plt.show()

# shows graph of average CV vs. average running count over time (hands) using partially trained models of partial fractions ns (averages are taken from nth hands of separate games)
def run_avg_partials(ns):
    ncols = 2
    fig,axes = plt.subplots(nrows = math.ceil(len(ns)/ncols),ncols=ncols)
    decks = 3
    times = 20
    models = []
    scaler = MinMaxScaler()

        # plot data from each partial model

    for n in ns:            
        models.append('cvmodel_partial'+str(int(n*100))+'.pkl')
    cnts,partials = sim.main(10,decks,count='decks',times=times,model=models)
    #cnts = cnts[0];partials=partials[0]
    

    # find the length of the longest game (number of hands)
    deck_lens = []
    for vec in cnts:
        deck_lens.append(len(vec))
    hands = max(deck_lens)
    mean_counts = []
    mean_ptls = []
    # manually average values (not all dimensions of these arrays are np.ndarrays, some are lists, and aren't of uniform length)
    for i in range(hands): # traverse through columns (hands)
        n = 0; sm = 0
        for j in range(len(cnts)): # traverse through rows (games)
            if i < len(cnts[j]): # if the index is within the vector
                sm+=cnts[j][i]
                n+=1
        mean_counts.append(sm/n) # append the average
        
    for j in range(len(ns)): # traverse through partials
        mean_ptls.append([])
        for i in range(hands): # traverse through hands (some games may not have an ith hand)
            n=0;sm=0
            for k in range(len(partials)): # traverse through games
                if len(partials[k][j]) > i: # if index is within the vector
                    sm+= partials[k][j][i]
                    n+=1
            mean_ptls[j].append(sm/n) # this creates a 2D array, rows being partials and columns being hands, values are averages over games
                
    alpha = 1/len(cnts) # we want the alpha of each line segment to be proportional to the number of games that have a hand of that index
    counts = scaler.fit_transform(np.array(mean_counts).reshape(-1,1))

    # plot data from each partial model
    for i,n in enumerate(ns):
        cvs = mean_ptls[i]
        cvs = scaler.fit_transform(np.array(cvs).reshape(-1,1))


        if ncols >1:
            ax = axes[i//ncols][i%ncols] # linear traversal of a 2D axes array
        else:
            ax = axes[i]
        ax.set_title(label='CV Model Trained and Tested with '+str(int(n*100))+'% of data')
        # plot many different segmented lines of small alphas that overlay to create a gradient, used to display how many different games are in the average from each line segment
        line,cnt = None,None
        for ln in deck_lens:
            line, = ax.plot(np.linspace(1,ln,ln),cvs[0:ln],alpha=alpha,c='blue',label='Cerebri Value')
            cnt, = ax.plot(np.linspace(1,ln,ln),counts[0:ln],alpha=alpha,label='Running Count',c='orange')
        line.set_alpha(1.0);cnt.set_alpha(1.0)


        ax.legend(handles=[line,cnt],labels=['Cerebri Value','Running Count'])
    plt.show()
    
# same as run_avg_partials except one line is plotted that is the difference between the CV and count
def run_avg_partials_difference(ns):
    ncols = 2
    fig,axes = plt.subplots(nrows = math.ceil(len(ns)/ncols),ncols=ncols)
    decks = 3
    times = 20
    models = []
    scaler = MinMaxScaler()

        # plot data from each partial model

    for n in ns:            
        models.append('cvmodel_partial'+str(int(n*100))+'.pkl')
    cnts,partials = sim.main(10,decks,count='decks',times=times,model=models)
    #cnts = cnts[0];partials=partials[0]
    

    # find the length of the longest game (number of hands)
    deck_lens = []
    for vec in cnts:
        deck_lens.append(len(vec))
    hands = max(deck_lens)
    mean_counts = []
    mean_ptls = []
    # manually average values (not all dimensions of these arrays are np.ndarrays, some are lists, and aren't of uniform length)
    for i in range(hands): # traverse through columns (hands)
        n = 0; sm = 0
        for j in range(len(cnts)): # traverse through rows (games)
            if i < len(cnts[j]): # if the index is within the vector
                sm+=cnts[j][i]
                n+=1
        mean_counts.append(sm/n) # append the average
        
    for j in range(len(ns)): # traverse through partials
        mean_ptls.append([])
        for i in range(hands): # traverse through hands (some games may not have an ith hand)
            n=0;sm=0
            for k in range(len(partials)): # traverse through games
                if len(partials[k][j]) > i: # if index is within the vector
                    sm+= partials[k][j][i]
                    n+=1
            mean_ptls[j].append(sm/n) # this creates a 2D array, rows being partials and columns being hands, values are averages over games
                
    alpha = 1/len(cnts) # we want the alpha of each line segment to be proportional to the number of games that have a hand of that index
    counts = scaler.fit_transform(np.array(mean_counts).reshape(-1,1))

    # plot data from each partial model
    for i,n in enumerate(ns):

        cvs = mean_ptls[i]
        cvs = scaler.fit_transform(np.array(cvs).reshape(-1,1))


        if ncols >1:
            ax = axes[i//ncols][i%ncols] # linear traversal of a 2D axes array
        else:
            ax = axes[i]
            
        ax.set_title(label='Difference between running count and '+str(int(n*100))+'% CV Model')
        
        # plot many different segmented lines of small alphas that overlay to create a gradient, used to display how many different games are in the average from each line segment
        for ln in deck_lens:
            ax.plot(np.linspace(1,ln,ln),counts[0:ln]-cvs[0:ln],alpha=alpha,c='blue') # we want to plot the difference each time

        if i==0: # we want all our plots to have the same scale, so use the scale of the first (and least accurate model) for the rest
            ylim=ax.get_ylim()
        else: ax.set_ylim(ylim)

        ax.axhline(0,color='black',lw=0.5) # line on the x-axis
        mean=np.mean(counts-cvs) # calculate the mean difference
        ax.axhline(mean,color='red',linestyle='--') # plot the mean
        ax.annotate('mean = '+str(round(mean,2))+' | std = '+str(round(np.std(counts-cvs),2)),xy=(hands*0.35,ax.get_ylim()[0]+0.1)) # annotate the mean


    plt.show()
        
        

#run_partials([0.1,0.3,0.5,0.8,1.0])
#run_avg_partials([0.1,0.3,0.5,0.8,1.0])
#run_avg_partials_difference([0.1,0.3,0.5,0.8,1.0])
#main()
    
