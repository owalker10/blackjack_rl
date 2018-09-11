import q_object_analysis as sim
from matplotlib import pyplot as plt
import numpy as np
import blackjack_object as bj
'''
Plot a timeseries graph of net gain from a Q-learning model vs. basic strategy

'''

def single_series(num_iters,q_table='bj_q_table1.csv',quantiles=[33,66],cvbet=False,nocv=False):
    '''
    Plots two lines, one from blackjack using basic strategy and the other using a trained Q-table
    Each line represents one game

    parameters:
        num_iters: number of hands per line
        q_table: .csv file to use for the trained Q-table
        quantiles: quantiles boundaries in terms of percents to use for the Cerebri Value bins (exclude 0 and 100)
        cvbet:
            == True: uses the calculated Cerebri Value to place bets
            == False: places a constant bet (arbitrality set to $100)
        nocv:
            == True: ignores the Cerebri Value when creating states (only True when using a basic strategy Q-table aka q_table = 'bj_q_table_bs.csv' and kind='q')
            == False: uses the Cerebri Value when creating states
    '''
    
    sim.change_quantiles(quantiles)
    
    q_data,q_bet = sim.main(q_table=q_table,cvbet=cvbet,nocv=nocv,kind='q',num_iters=10000)
    bs_data,bs_bet = sim.main(kind='bs',num_iters=10000)

    q_data = [sum(q_data[:i+1]) for i in range(len(q_data))]
    bs_data = [sum(bs_data[:i+1]) for i in range(len(bs_data))]

    q_percent = np.array(q_data)/sum(q_bet)*100
    bs_percent = np.array(bs_data)/sum(bs_bet)*100

    print(str(q_percent[-1])+'%',str(bs_percent[-1])+'%')

    # format the graph: add titles and labels
    ax = plt.gca()

    ax.set_title('Net Gain over Time')
    ax.set_ylabel('Percent of Final Bet Returned')
    ax.set_xlabel('Hands Played')

    ax.axhline(0,linestyle='--',color='black')

    ax.annotate(' '+str(round(q_percent[-1],3))+'%',(len(q_data),q_percent[-1]))
    ax.annotate(' '+str(round(bs_percent[-1],3))+'%',(len(bs_data),bs_percent[-1]))
    
    ax.plot(q_percent,label='Q-Learning')
    ax.plot(bs_percent,label='Basic Strategy')

    locs,labels = plt.xticks()
    if locs[0] != 0: locs=locs[1:]
    np.append(locs,locs[-1]+locs[1]-locs[0])
    labs = [str(int(loc)) for loc in list(locs)]
    labs[-1] = ''
    plt.xticks(locs,labs)
    
    locs,labels = plt.yticks()
    labs = [str(loc)+'%' for loc in locs]
    plt.yticks(locs,labs)
    
    ax.legend()
    
    plt.show()

def avg_series(size,num_iters,q_table='bj_q_table1.csv',quantiles=[33,66],cvbet=False,nocv=False):
    sim.change_quantiles(quantiles)

    # simulate data and format it as a cumulative sum for each time increment in percent units
    q_data = []
    bs_data = []
    for n in range(size):
        q,bet = sim.main(kind='q',num_iters=num_iters,cvbet=cvbet,q_table=q_table,nocv=nocv)
        q = [sum(q[:i+1]) for i in range(len(q))]
        q = np.array(q)/sum(bet)*100

        bs,bet = sim.main(kind='bs',num_iters=num_iters)
        bs = [sum(bs[:i+1]) for i in range(len(bs))]
        bs = np.array(bs)/sum(bet)*100

        q_data.append(q)
        bs_data.append(bs)

    # average all the games together to make one line
    q_data = np.mean(np.array(q_data),axis=0)
    bs_data = np.mean(np.array(bs_data),axis=0)

    print(str(q_data[-1])+'%',str(bs_data[-1])+'%')

    # format the graph: add titles and labels
    ax = plt.gca()

    ax.set_title('Net Gain over Time, Average of '+str(size)+' Games')
    ax.set_ylabel('Percent of Final Bet Returned')
    ax.set_xlabel('Hands Played')

    ax.axhline(0,linestyle='--',color='black')

    ax.annotate(' '+str(round(q_data[-1],3))+'%',(len(q_data),q_data[-1]))
    ax.annotate(' '+str(round(bs_data[-1],3))+'%',(len(bs_data),bs_data[-1]))
    
    ax.plot(q_data,label='Q-Learning with CV-adjusted Betting')
    ax.plot(bs_data,label='Basic Strategy')

    locs,labels = plt.xticks()
    if locs[0] != 0: locs=locs[1:]
    np.append(locs,locs[-1]+locs[1]-locs[0])
    labs = [str(int(loc)) for loc in list(locs)]
    labs[-1] = ''
    plt.xticks(locs,labs)
    
    locs,labels = plt.yticks()
    labs = [str(round(loc,2))+'%' for loc in locs]
    plt.yticks(locs,labs)
    
    ax.legend()
    
    plt.show()

# same as avg_series but with multiple Q-table lines
def mult_q_series(tables,qs,legend=[],nocvlist=None,cvbetlist=None):
    size = 1000 # number of simulations to average over
    num_iters = 5000 # length of each game

    # simulate data and format it as a cumulative sum for each time increment in percent units
    
    bs_data = []

    ax = plt.gca()

    rules = bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False)

    # simulate all the Q-learning model data
    for j,quantiles_and_table in enumerate(zip(qs,tables)):
        quantiles,table = quantiles_and_table
        if nocvlist is not None: nocv = nocvlist[j]
        else: nocv = False
        if cvbetlist is not None: cvbet = cvbetlist[j]
        else: cvbet = False
        q_data = []
        sim.change_quantiles(quantiles)
        
        
        for n in range(size):
            q,bet = sim.main(kind='q',num_iters=num_iters,q_table=table,nocv=nocv,cvbet=cvbet,rules=rules)
            q = [sum(q[:i+1]) for i in range(len(q))]
            q = np.array(q)/sum(bet)*100

            q_data.append(q)

        q_data = np.mean(np.array(q_data),axis=0)
        
        ax.plot(q_data,label=legend[j])
        ax.annotate(' '+str(round(q_data[-1],3))+'%',(len(q_data),q_data[-1]))
        print(j+1,'of',len(qs),'finished')

    # simulate the basic strategy data
    for n in range(size):
        bs,bet = sim.main(kind='bs',num_iters=num_iters,rules=rules)
        bs = [sum(bs[:i+1]) for i in range(len(bs))]
        bs = np.array(bs)/sum(bet)*100
        bs_data.append(bs)
    print('bs finished')

        
    bs_data = np.mean(np.array(bs_data),axis=0)


    # format the graph: add titles and labels
    

    ax.set_title('Net Gain over Time, Average of '+str(size)+' Games')
    ax.set_ylabel('Percent of Final Bet Returned')
    ax.set_xlabel('Hands Played')

    ax.axhline(0,linestyle='--',color='black')

    ax.annotate(' '+str(round(bs_data[-1],3))+'%',(len(bs_data),bs_data[-1]))
    
    ax.plot(bs_data,label='Internet Basic Strategy')

    locs,labels = plt.xticks()
    if locs[0] != 0: locs=locs[1:]
    np.append(locs,locs[-1]+locs[1]-locs[0])
    labs = [str(int(loc)) for loc in list(locs)]
    labs[-1] = ''
    plt.xticks(locs,labs)
    
    locs,labels = plt.yticks()
    labs = [str(round(loc,2))+'%' for loc in locs]
    plt.yticks(locs,labs)

    ax.legend()
    
    plt.show()    
                

if __name__ == '__main__':
    #single_series(10000)
    #avg_series(1000,1000,q_table='bj_q_table10.csv',quantiles=[33,66])
    #mult_q_series(['bj_q_table1.csv','bj_q_table_bs.csv'],[[33,66],[33,66]],legend=['CV-based Q-learning with CV Betting','Basic Strategy Q-learning'],nocvlist=[False,True])
