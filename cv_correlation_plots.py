import pandas as pd
from matplotlib import pyplot as plt

# plot an average returns vs. Cerebri Value graph with separate lines for each action
def main():
    data = pd.read_csv('bj_q_table1.csv',index_col=0)

    bins = [0,1,2]
    df = pd.DataFrame(index = bins, columns = ['stand','hit','double down','split'])

    for b in bins:

        rows = data[data.index.str.find('cv bin: '+str(b)) != -1]
        
        avgs = rows.mean(axis=0) # average all the rows together, leaving the different actions as separate elements
        df.loc[b,:] = avgs.loc[df.columns] # put these values into the row with the correct CV bin

    ax = plt.gca()

    for col in df.columns:
        plt.plot(df[col],label=col)
        
    ax.legend()
    ax.set_title('Returns vs. CV by Decision Type')
    ax.set_xlabel('CV bin (less than or equal to)')
    ax.set_ylabel('Average Returns')
    loc,_ = plt.xticks()
    plt.xticks([n for n in range(0,len(bins))],['33%','66%','100%'])

    plt.show()
        

        

if __name__ == '__main__':
    main()
