import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde as kde
import numpy as np
from sklearn.externals import joblib
import math

# using a series of discard piles, visualize distribution of the percent composition of given card in the discard pile
def main():
    data = pd.read_csv('discard_data_1000000.csv').iloc[:,2:]

    def card_dist(card,sample_size):
        sample = data[:sample_size]
        dist = sample[card]/sample.sum(axis=1)
        dist.fillna(value=0.0,inplace=True)
        
        return dist

    # plot a PDF of the distribution
    def plot_pdf(card,sample_size):
        dist = card_dist(card,size)
        pdf = kde(dist)
        xs = np.linspace(dist.min(),dist.max(),100)
        ys = pdf(xs)

        mean = dist.mean()
        
        ax = plt.gca()
        ax.fill_between(xs,ys)
        ax.axvline(mean,color='red',linestyle='--')
        ax.annotate(s=' mean = '+str(round(mean,2)),xy=(mean,ys.max()/2))
        ax.set_title('Frequency of '+str(card)+' from '+str(size)+' discard piles')
        plt.show()
        
    # plot a bar plot of the distribution
    def plot_bar(card,sample_size):
        dist = card_dist(card,size)
        bars = sorted(list(set(list(dist))))
        print(bars)
        freq = []
        for bar in bars:
            freq.append(dist[dist==bar].count()/sample_size)

        ax = plt.gca()
        mean = dist.mean()
        ax.bar(bars,freq,width=0.014)
        ax.axvline(mean,color='red',linestyle='--')
        ax.set_title('Frequency of '+str(card)+' from '+str(size)+' discard piles')
        plt.show()
        
    # bin the set of finite frequencies and plot a bar plot where each bin is a bar
    def plot_bin_bar(card,sample_size,bins):
        dist = card_dist(card,size)
        bins[-1] = (bins[-1][0],dist.max())
        bars = sorted(list(set(list(dist))))
        freq = []
        widths = []
        xs = []
        
        for lower,upper in bins:
            freq.append(dist[(dist>=lower) & (dist<upper)].count()/sample_size)
            widths.append(upper-lower)
            xs.append(lower+(upper-lower)/2)
            
        widths[0] = 0.025

        ax = plt.gca()
        ax.bar(xs,freq,width=widths,edgecolor='black')
        ax.set_title('Frequency of '+str(card)+' from '+str(size)+' discard piles, put into '+str(len(bins))+' bins')
        plt.show()

    def plot_bin_result_bar(card,sample_size,bins=None,quantiles=None):
        results = pd.read_csv('discard_data_1000000.csv').iloc[:sample_size,1]

        dist = card_dist(card,size)
        
        if quantiles is not None:
            qs = np.percentile(dist,quantiles)
            print(qs[:])
            bins = [(dist.min(),qs[0])]
            for i in range(len(qs)-1):
                bins.append((qs[i],qs[i+1]))
            bins.append((qs[-1],dist.max()))

        bins[0] = (dist.min(),bins[0][1]); bins[-1] = (bins[-1][0],dist.max())


        avg_result = []
        xs = []
        widths = []

        for lower,upper in bins:
            bin_results = results[(dist>=lower) & (dist<upper)]
            avg_result.append(bin_results.mean())
            xs.append(lower+(upper-lower)/2)
            widths.append(upper-lower)
        
        widths[0] = 0.025
         
        ax = plt.gca()
        ax.bar(xs,avg_result,width=widths,edgecolor='black')
        ax.set_title('Average result vs binned probability '+str(card)+' from '+str(size)+' discard piles, put into '+str(len(bins))+' bins')
        ax.set_xlabel('Percent composistion of '+str(card)+' in discard pile')
        ax.set_ylabel('Average result (1 for win, 0 for loss)')
        plt.show()

    def plot_grid_result_bars(cards,sample_size,bins=None,quantiles=None):
        results = pd.read_csv('discard_data_1000000.csv').iloc[:sample_size,1]
        ncols=2
        fig,axes = plt.subplots(nrows=math.ceil(len(cards)/ncols),ncols=ncols)
        plt.suptitle('Avg Result vs binned discard frequencies with '+str(sample_size)+' samples')
        for n,card in enumerate(cards):
            dist = card_dist(card,size)
        
            if quantiles is not None:
                qs = np.percentile(dist,quantiles)
                #print(qs[:])
                bins = [(dist.min(),qs[0])]
                for i in range(len(qs)-1):
                    bins.append((qs[i],qs[i+1]))
                bins.append((qs[-1],dist.max()))

            bins[0] = (dist.min(),bins[0][1]); bins[-1] = (bins[-1][0],dist.max())


            avg_result = []
            xs = []
            widths = []

            for lower,upper in bins:
                bin_results = results[(dist>=lower) & (dist<=upper)]
                avg_result.append(bin_results.mean())
                xs.append(lower+(upper-lower)/2)
                widths.append(upper-lower)
            
            
            widths[0] = 0.025
             
            ax = axes[n//ncols][n%ncols]
            
            ax.set_title(card)
            ax.set_xlabel('Percent composistion')
            ax.set_ylabel('Average result')
            ax.set_ylim(bottom=np.amin(avg_result)-0.001)
            ax.set_ybound(upper=np.amax(avg_result)+0.02)
            ax.bar(xs,avg_result,width=widths,edgecolor='black')
        plt.show()
            

    def plot_result_curve(card,sample_size):
        results = pd.read_csv('discard_data_1000000.csv').iloc[:sample_size,1]
        dist = card_dist(card,size)
        dist_set = sorted(list(set(dist)))

        avg_results = []
        
        for percent in dist_set:
            percent_results = results[dist == percent]
            avg_results.append(percent_results.mean())

        for percent,result in zip(dist_set,avg_results):
            print(percent,result,dist[dist==percent].count())

        ax = plt.gca()
        ax.plot(dist_set,avg_results)
        ax.set_title('Average result vs discard pile frequency of '+str(card)+' from '+str(size)+' discard piles')
        ax.set_xlabel('Percent composistion of '+str(card)+' in discard pile')
        ax.set_ylabel('Average result (1 for win, 0 for loss)')
        plt.show()

    def plot_mult_result_curve(cards,sample_size):
        results = pd.read_csv('discard_data_1000000.csv').iloc[:sample_size,1]
        ax = plt.gca()
        for card in cards:
            dist = card_dist(card,size)
            dist_set = sorted(list(set(dist)))

            avg_results = []
            
            for percent in dist_set:
                percent_results = results[dist == percent]
                avg_results.append(percent_results.mean())

            ax.plot(dist_set,avg_results,label=card)


        ax.set_title('Average result vs discard pile frequency of '+str(cards)+' from '+str(size)+' discard piles')
        ax.set_xlabel('Percent composition of '+str(cards)+' in discard pile')
        ax.set_ylabel('Average result (1 for win, 0 for loss)')
        ax.legend()
        plt.show()

    def plot_cv_dist(sample_size):
        model = joblib.load('cvmodel.pkl')
        cvs = model.predict_proba(data[:sample_size]).T[1]
        pdf = kde(cvs)
        xs = np.linspace(cvs.min(),cvs.max(),1000)
        ys = pdf(xs)

        ax = plt.gca()
        ax.plot(xs,ys)
        ax.set_title('Cerebri Value distribution over '+str(sample_size)+' discard piles')

        plt.show()

    def plot_result_cv_bins(sample_size,bins=None,quantiles=None):
        results = pd.read_csv('discard_data_1000000.csv').iloc[:sample_size,1]
        model = joblib.load('cvmodel.pkl')
        cvs = model.predict_proba(data.iloc[:sample_size,:]).T[1]
        cvs = pd.Series(cvs)
        if quantiles is not None:
            qs = np.percentile(cvs,quantiles)
            print(qs[:])
            bins = [(cvs.min(),qs[0])]
            for i in range(len(qs)-1):
                bins.append((qs[i],qs[i+1]))
            bins.append((qs[-1],cvs.max()))

        bins[0] = (cvs.min(),bins[0][1]); bins[-1] = (bins[-1][0],cvs.max())

        xs=[]
        avg_result=[]
        widths=[]

        for lower,upper in bins:
            bin_results = results[(cvs>=lower) & (cvs<upper)]
            avg_result.append(bin_results.mean())
            xs.append(lower+(upper-lower)/2)
            widths.append(upper-lower)

        ax=plt.gca()
        ax.bar(xs,avg_result,width=widths,edgecolor='black')
        ax.set_title('Average result vs binned Cerebri Values over '+str(sample_size)+' discard piles')
        ax.set_xlabel('Raw Cerebri Value')
        ax.set_ylabel('Average Result (0 for loss, 1 for win)')
        plt.show()
        
    
        

    card = '10'
    size = 1000000
    
    #plot_pdf(card,size)
    #plot_bar(card,size)

    cards = ['2','3','4']
    #plot_mult_result_curve(cards,size)

    #plot_result_curve(card,size)

    bins = [(0,0.00001),(0.00001,0.1),(0.1,1.0)]
    
    #plot_bin_bar(card,size,bins)
    #quantiles=[10,33,66,90]
    #quantiles=[2,5,10,20,30,40,50,60,70,80,90,95,98]
    #plot_bin_result_bar(card,size,quantiles=quantiles)
    cards=['2','3','4','5','6','7','8','9','10','J','Q','K','A']
    #plot_grid_result_bars(cards,size,quantiles=quantiles)

    #plot_cv_dist(size)
    quantiles = [33,66]
    plot_result_cv_bins(size,quantiles=quantiles)


if __name__ == '__main__':
    main()
