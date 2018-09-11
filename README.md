# blackjack_rl

This is a repository of work from my summer internship. The premise is to demonstrate how adding engineered features to reinforcement learning can significantly increase performance. The example here is blackjack. The "Cerebri Value" being referred to is the engineered feature, named after the product that Cerebri AI calculates for its customers.

## Key Terms
- upcard: The dealer is initially dealt two cards, one face up and one face down. The upcard is the dealer's face-up card.
- Basic Strategy: A set of logic that determines that statistically best blackjack move to make for any given set of player cards and dealer upcard.
- hand: A hand either refers to the set of cards one player or one dealer has at a time, or a full round of blackjack starting with bets and ending with payout.

### cards.py

This module contains methods and objects related to blackjack play including a `Card` object, as well as `Player`, `Deck`, and `Hand`. This module is used by any script that simulates a blackjack game. You never need to run this script.

### basicstrategy.py


This module reads a table from basic_strategy.csv. Its `get_move()` method takes the player hand and dealer upcard as parameters, matches them to the basic strategy table, and returns a string representing the correct move to be made. `basic_strategy.get_move` is used by any script that simulates a blackjack game being played with basic strategy. You never need to run this script.

## Simulation Scripts

The old simulation scripts have been replaced by one that uses `blackjack_object.py`, which holds an interactable blackjack object

- q_object_analysis.py: a script that is utilized by other scripts but can also be run itself. Uses blackjack_object and takes several optional parameters. The output of its `main()` function are two variables, `won` and `bet`
	- `won`: the total sum of money won from the simulated hands
	- `bet`: the total amount of money bet during the simulated hands
- `sum(won)/sum(bet)` will give the fractional value of money returned
- the parameters of `main()` are as follows:
	- `q_table`: string representing the .csv file name of the Q-table to be used
	- `num_iters`: number of hands to be played
	- `kind`: inputting "q" will cause the method to use the Q-table that it was given, inputting "bs" will cause the method to use `basicstrategy.py` to make moves based on a basic strategy table
	- `cvbet`: when False, the simulation places a constant, arbitrary bet (set to $100) every hand; when set to True, the simulation will made Cerebri Value adjusted bets, which significantly increases percent returns
	- `nocv`: when False, the simulation will use the Cerebri Value when creating States; this should only be True when you're using the basic strategy Q-table to perform simulations (`q_table='bj_q_table_bs.csv,kind='q',nocv=True`)
	- `rules`: a `Rules` object from `blackjack_object.py`, holds multiple values representing individual rules; the blackjack simulation will play according to the rules specified
		- example: `rules=bj.Rules(numdecks=2,double_after_split=True,hit_split_aces=False,resplit_aces=False,cut=0.5,late_surrender=False,h17=True)`
	- `onehand`: when None, the simulation plays normally; otherwise, `onehand` should be in the form of [[player card 1, player card2], dealer upcard] and will cause the simulation to play repeated hands that always start with the given player hand and dealer upcard
		- example: `onehand=[['A','2'],'9']`

## ML Models

All of these scripts use data of some format to train a machine learning model either to predict or play

- cv_model.py: trains a logistic regression classifier to predict wins or losses based only on the contents of the discard pile. The probability function of this trained model is our raw Cerebri Value

- tictactoe_rl.py: trains a Q-learning model to play tic-tac-toe where the opponent plays in a pre-determined and stochastic manner (this was a warmup for writing Q-learning models)

- blackjack_rl.py: trains a Q-learning model to play blackjack using on-policy Monte Carlo control with a Q-table (this is currently the highest performing blackjack RL model)

- blackjack_rl_bs.py: similar to blackjack_rl but doesn't use a Cerebri Value as part of its training. The trained Q-table is effectively a Basic Strategy table

## Visualization

Uses pyplot to visualize data

- basic_data_analysis.py: simulates a large number of blackjack hands according to basic strategy and creates a histogram and PDF of the net returns in terms of percentages

- basic_data_analysis_onehand.py: contains several methods to visualize data and distributions of simulated blackjack data from only one table combination (same two starting player cards and same dealer upcard, hole card and deck differ)

- card_distributions.py: several methods for visualizing the distributions of data such as frequencies of a given card in the discard pile, and the average returns vs binned Cerebri Values

- cv_bet_analysis.py: contains several methods used to visualize net gain/loss by betting with Cerebri Values vs. betting a fixed amount

- cv_correlation_plots.py: plot a net gain/loss vs Cerebri Value graph where each action is a separate line, used to visualize trends in expected result as the Cerebri Value changes

- cv_vs_count_demo.py: several methods to visualize the change in Cerebri Value and card count "running count" over time to compare the similarity of the two

- q_object_analysis.py: when ran as the main script, prints net returns of a simulation according to the call of 'main()'

- q_vs_bs_timeseries.py: several methods for plotting percent gain vs. time of basic strategy and Q-learning models

## Trained Models

These files are the output of an ML model training and can be used as a guide for decision making or loaded as a model into another script

- cvmodel.pkl: Cerebri Value model. Trained to predict wins and losses using only the contents of the discard pile. Can be loaded into another script with model = joblib.load('cvmodel.pkl')

- cvmodel_partial{n}.pkl: CV model trained and tested on only a fraction of the discard pile where n is the percentage of the discard pile that is allowed to be seen

- bj_q_table(...).csv: a series of .csv files that are trained Q-tables from Q-learning trains of varying iterations, methods, and number of Cerebri Value bins
	- bj_q_table1.csv: Q-table trained on 20 million iterations, 3 CV bins, and a c value of 150
  	- bj_q_table1-2.csv: Q-table trained on 50 million iterations, 3 CV bins, and a c value of 150
	- bj_q_table1-3.csv: Q-table trained on 100 million iterations, 7 CV bins, and a c value of 150
	- bj_q_table_bs.csv: a Q-table that was trained without Cerebri Values, and so functions the same as a basic strategy table

## Others

- create_csv_discard.py: simulates blackjack games and takes the discard piles at the beginning of every hand and saves the data to a .csv

- odds.csv: contains playing odds for combinations of player hands and dealer upcards; used as comparison data for "one hand" analysis

- blackjack_object.py: contains several objects used in simulations
	- `class Blackjack()`: an object representing a game of blackjack, and is used by simulation scripts
	- `class Controlled_Blackjack()`: CURRENTLY OUTDATED, DO NOT USE; was previously used in a controlled off-policy Q-learning script to create blackjack games with specific card combinations; said off-policy Q-learning proved to be ineffective
	- `class State()`: represents a "state" of blackjack, used as individual row indexes in Q-tables
	- `class Rules()`: represents a set of blackjack rules
		- `numdecks`: the number of decks that are played with, default 1
		- `double_after_split`: whether the player can double down after splitting hands, default True
		- `hit_split_aces`: whether the player can take a third card on a hand created by splitting aces, deault True
		- `resplit_aces`: whether the player can resplit aces that have already been split once, default True
		- `split_hands`: maximum number of hands the player can have by splitting, default 4
		- `late_surrender`: whether the player can surrender their hand, losing half their bet, default True
		- `bj_payout`: fraction of bet returned when the player has a natural blackjack, default 3/2
		- `cut`: placement of a cut card, in units of decks (i.e. if `cut` = 0.5, the discard pile will be re-shuffled back in when the deck reaches 26 cards), default None (aka 0)
		- `h17` whether the dealer hits on a soft 17 (ace plus a total sum of 6), default True
