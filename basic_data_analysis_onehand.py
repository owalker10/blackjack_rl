import q_object_analysis as sim
import blackjack_object as bj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde as kde
import cards
import random

'''
This script simulates blackjack hands for a single combination of two starting player cards and a dealer upcard,
and generates Kernel Density Estimations and compares the calculated values to those found on wizardofodds.com

Note: the dealer hole card is NOT HELD CONSTANT, and is the only variable item between hands

Go to bottom for calling methods!
'''

rules=bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False)



def plot_pdf(ph,up,real,limit,times):
    '''
    given parameters, generate a graph of a probability densitity function of the percent returns of many blackjack games using Kernel Density Estimation
    supports .show() or .savefig(), just uncomment the call you want!

        parameters:
            ph: list of Card objects representing the player hand
            up: Card object representing the  dealer's upcard
            real: calculated return percentage via Wizard of Odds
            limit: numbers of hands to be played per each data point
            times: number of data points to be collected
    '''

    fig = plt.figure()
    plt.subplots_adjust(hspace=0.35)

   
    data = []

    # generate all the data
    for n in range(times):
        total_won,total_bet = sim.main(num_iters=limit,kind='bs',rules=rules,onehand=[ph,up])
        data.append(sum(total_won)/(100*limit))

    data = np.asarray(data)
    percent = data * 100

    ax = fig.gca()
    ax.set_title('KDE of percentage return with hand '+str(ph[0].num)+str(ph[1].num)+' and dealer upcard '+up.num)
    xs = np.linspace(percent.min(),percent.max(),100)
    pdf = kde(percent); ax.plot(xs,pdf(xs))
    mean = np.mean(percent)
    ax.axvline(x=real,color='yellow',label='Wizard of Odds value')
    ax.annotate(' real value = '+str(round(real,3))+'%',(real,pdf(xs).max()/2))
    ax.axvline(x=mean,color='red',linestyle='--',label='mean simulation value')
    ax.annotate(' mean = '+str(round(mean,3))+'%',(mean,pdf(xs).max()/4))
    ax.legend()
    plt.show() # UNCOMMENT ME FOR PLOT WINDOWS
    #fig.savefig('onehand_pdfs/'+ph[0].num+ph[1].num+up.num+'-'+str(times)+'.png') # UNCOMMENT ME TO SAVE FIGS AS PNG
    

odds = pd.read_csv('odds.csv')
odds['odds'] = odds['odds']*100


def random_samples(rand):
    '''
    randomly picks "rand" card combinations from the table and displays their return distribution PDF's
    '''
    global odds
    rows = []
    for x in range(rand):
        rows.append(np.random.randint(odds.shape[0]))
    sample = odds.iloc[rows,:].reset_index(drop=True)
    for x in range(sample.shape[0]):
        nums = sample['hand'].iloc[x].split(',')
        player_hand = [cards.Card(nums[0],'spades'),cards.Card(nums[1],'diamonds')]
        upcard = cards.Card(str(sample['upcard'].iloc[x]),'hearts')
        adv = float(sample['odds'].iloc[x])
        plot_pdf(player_hand,upcard,adv,20,100000)

def card_pdf(nums,up):
    '''
    given strings that represent a card combination, create the Cards and pass them to plot_pdf to produce the PDF
    '''
    global odds
    player_hand = [cards.Card(nums[0],'spades'),cards.Card(nums[1],'diamonds')]
    upcard = cards.Card(up,'hearts')
    match = nums[0]+','+nums[1]
    adv = odds[(odds['hand']==match)&(odds['upcard'].apply(lambda x:str(x))==up)];adv=adv['odds'].iloc[0]
    plot_pdf(player_hand,upcard,adv,10,10000)


def rand_correlation(rand,limit,times):
    '''
    find the correlation coefficient between randomly sampled simulated data returns vs an external dataset's calculated returns

    parameters:
        rand: number of card combinations to randomly sample
        limit: number of hands per game
        times: number of games per simulation (1 game is 1 datapoint)
    
    returns: correlation coefficient
    '''
    global odds
    sims = []
    reals = []
    rows = random.sample(range(0, odds.shape[0]), rand)

    sample = odds.iloc[rows,:].reset_index(drop=True)

    for x in range(sample.shape[0]):
        nums = sample['hand'].iloc[x].split(',')
        player_hand = [cards.Card(nums[0],'spades'),cards.Card(nums[1],'diamonds')]
        upcard = cards.Card(str(sample['upcard'].iloc[x]),'hearts')
        adv = float(sample['odds'].iloc[x])

        data = []
        
        for n in range(times):
            total_won,total_bet = sim.main(num_iters=limit,kind='bs',rules=rules,onehand=[player_hand,upcard])
            data.append(sum(total_won)/(100*limit))

        data = np.asarray(data)
        returns = data*100
        mean = np.mean(returns)
        sims.append(mean)
        reals.append(adv)
    r = np.corrcoef(sims,y=reals)
    return r[0][1]


def find_outliers(limit,times,error):
    '''
    runs through every card combination, simluating average returns and comparing them to the external values
    prints the combination if the different between the simulated average and the externally calculated one exceeds <error> %

    parameters:
        limit: number of hands per game
        times: number of games per simulation (1 game is 1 datapoint)
        error: if |expected_returns - calculated_returns| > error, print out the card combination and values

    note: simulated blackjack data is very variable, so using simulated means from small amounts of trials may not accurately represent the true mean
    '''
    global odds
    for x in range(odds.shape[0]):
        nums = odds['hand'].iloc[x].split(',')
        player_hand = [cards.Card(nums[0],'spades'),cards.Card(nums[1],'diamonds')]
        upcard = cards.Card(str(odds['upcard'].iloc[x]),'hearts')
        adv = float(odds['odds'].iloc[x])

        data = []
        
        for n in range(times):
            total_won,total_bet = sim.main(num_iters=limit,kind='bs',rules=rules,onehand=[player_hand,upcard])
            data.append(sum(total_won)/(100*limit))


        data = np.asarray(data)
        returns = data*100
        mean = np.mean(returns)
        if abs(mean-adv)>error:
            print(nums[0],nums[1],'and',upcard.num,str(abs(mean-adv))+'%','real: ',adv,'simulated: ',mean)
    

#card_pdf(['9','2'],'3') # UNCOMMENT ME FOR SPECIFIC COMBINATIONS
#random_samples(5) # UNCOMMENT ME FOR RANDOM SAMPLES
#print(rand_correlation(int(odds.shape[0]),10,1000)) # UNCOMMENT ME FOR THE TOTAL CORRELATION COEFFICIENT OF RANDOM SAMPLES
#find_outliers(10,1000,3) # UNCOMMENT ME TO FIND OUTLIERS
    


