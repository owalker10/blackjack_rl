import q_object_analysis as sim
import blackjack_object as bj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde as kde
import random

'''
Compare distribution of blackjack returns with constant betting and CV-adjusted betting
'''

rules = bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False)



def two_plots():
    '''
    Display two separate distributions
    '''
    
    # CHANGE ME IF YOU WANT (limit is number of hands played per game, times in number of games played: 1 game = 1 data point)
    bet,limit,times = 10,30,500000

    # create a subplot figure of height 2 and width 1
    fig, axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.35)

    data = []
    #generate a series of net money gained/lossed over several hands
    for n in range(times):
        total_won,total_bet = sim.main(num_iters=limit,kind='bs',rules=rules)
        data.append(sum(total_won)/sum(total_bet))

    data = np.asarray(data) # so we can perform math
    percent = data * 100 # calculate percent money returned

    data = []
    #generate a series of net money gained/lossed over several hands
    for n in range(times):
        total_won,total_bet = sim.main(q_table='bj_q_table_bs.csv',num_iters=limit,kind='q',nocv=True,cvbet=True,rules=rules)
        data.append(sum(total_won)/sum(total_bet))

    data = np.asarray(data) # so we can perform math
    cv_percent = data * 100 # calculate percent money returned

    # plotting the constant bet KDE
    ax = axes[0]
    ax.set_title('KDE of Percent Returns with Constant Bet of $'+str(bet))
    xs = np.linspace(percent.min(),percent.max(),100)
    pdf = kde(percent); ax.plot(xs,pdf(xs))
    mean = np.mean(percent)
    ax.axvline(x=round(mean,3),color='red',linestyle='--') # graph the simulated mean as a vertical line
    ax.annotate('mean = '+str(round(mean,3))+'%',(mean,0))

    # plotting the CV bet KDE
    ax = axes[1]
    ax.set_title('KDE of Percent Returns with CV-adjusted Bet')
    xs = np.linspace(cv_percent.min(),cv_percent.max(),100)
    pdf = kde(cv_percent); ax.plot(xs,pdf(xs))
    mean = np.mean(cv_percent)
    ax.axvline(x=round(mean,3),color='red',linestyle='--') # graph the simulated mean as a vertical line
    ax.annotate('mean = '+str(round(mean,3))+'%',(mean,0))

    plt.show()

def overlap_plot():
    '''
    Display two overlapping distributions on the same graph
    '''
    
    # CHANGE ME IF YOU WANT (limit is number of hands played per game, times in number of games played: 1 game = 1 data point)
    limit,times = 30,50000
    
    data = []
    #generate a series of net money gained/lossed over several hands
    for n in range(times):
        total_won,total_bet = sim.main(num_iters=limit,kind='bs',rules=rules)
        data.append(sum(total_won)/sum(total_bet))

    data = np.asarray(data) # so we can perform math
    percent = data * 100 # calculate percent money returned

    data = []
    #generate a series of net money gained/lossed over several hands
    for n in range(times):
        total_won,total_bet = sim.main(q_table='bj_q_table_bs.csv',num_iters=limit,kind='q',nocv=True,cvbet=True,rules=rules)
        data.append(sum(total_won)/sum(total_bet))

    data = np.asarray(data) # so we can perform math
    cv_percent = data * 100 # calculate percent money returned



    # plotting the constant bet KDE
    ax = plt.gca()
    ax.set_title('KDE of Percent Returns with Constant Bet and CV-adjusted Bet')
    xs = np.linspace(percent.min(),percent.max(),100)
    pdf = kde(percent); ys= pdf(xs)
    ax.fill_between(x=xs,y1=ys,y2=0*len(ys),alpha=0.3,color='green',label='Constant Bet')
    mean = np.mean(percent)
    ax.axvline(x=round(mean,3),color='#5cc45e',linestyle='--') # graph the simulated mean as a vertical line
    ax.annotate(' constant mean = '+str(round(mean,3))+'%',(mean,0))

    # plotting the CV bet KDE
    xs = np.linspace(cv_percent.min(),cv_percent.max(),100)
    pdf = kde(cv_percent); ys=pdf(xs)
    ax.fill_between(x=xs,y1=ys,y2=0*len(ys),alpha=0.3,color='blue',label='CV Bet')
    mean = np.mean(cv_percent)
    ax.axvline(x=round(mean,3),color='#86b0f4',linestyle='--') # graph the simulated mean as a vertical line
    ax.annotate(' cv mean = '+str(round(mean,3))+'%',(mean,ax.get_ylim()[1]/3))

    ax.legend()

    plt.show()


#two_plots()
#overlap_plot()


    

