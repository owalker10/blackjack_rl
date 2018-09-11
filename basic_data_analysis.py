import q_object_analysis as sim
import blackjack_object as bj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde as kde

'''
This script simulates blackjack games using basic strategy and displays the distributions of net returns
'''

# CHANGE ME IF YOU WANT (limit is number of hands played per game, times in number of games played: 1 game = 1 data point)
limit,times = 30,100000
# rules for this game of blackjack, if you change the rules you should make sure you have the appropriate basic strategy table
rules = bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False)

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

# plotting the Kernel Density Estimation
ax = axes[0]
ax.set_title('KDE of percent gain/loss from blackjack games of '+str(limit)+' hands each')
xs = np.linspace(percent.min(),percent.max(),100)
pdf = kde(percent); ax.plot(xs,pdf(xs))
mean = np.mean(percent)
ax.axvline(x=round(mean,3),color='red',linestyle='--') # graph the simulated mean as a vertical line
ax.annotate('mean = '+str(round(mean,3))+'%',(mean,0))

# plotting the histogram
ax = axes[1]
ax.set_title('Histogram of percent gain/loss from blackjack games of '+str(limit)+' hands each')
ax.hist(percent,bins=35)
ax.axvline(x=round(mean,3),color='red',linestyle='--') # graph the simulated mean as a vertical line
ax.annotate('mean = '+str(round(mean,3))+'%',(mean,0))


plt.show()


