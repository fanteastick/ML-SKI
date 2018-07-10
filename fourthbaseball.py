import pandas as pd

#importing all the data from CSV files
master_df = pd.read_csv('People.csv', usecols=['playerID', 'nameFirst', 'nameLast', 'bats', 'throws', 'debut', 'finalGame'])
fielding_df = pd.read_csv('Fielding.csv',usecols=['playerID','yearID','stint','teamID','lgID','POS','G','GS','InnOuts','PO','A','E','DP'])
batting_df = pd.read_csv('Batting.csv')
awards_df = pd.read_csv('AwardsPlayers.csv', usecols=['playerID','awardID','yearID'])
allstar_df = pd.read_csv('AllstarFull.csv', usecols=['playerID','yearID'])
hof_df = pd.read_csv('HallOfFame.csv',usecols=['playerID','yearid','votedBy','needed_note','inducted','category'])
appearances_df = pd.read_csv('Appearances.csv')




##################################DATA CLEANING AND PREPROCESSING##################################

	#start w/ batting_df organizing
player_stats = {}
years_played={}

		# Create dictionaries for player stats and years played from `batting_df`
for i, row in batting_df.iterrows():
	playerID = row['playerID']
	if playerID in player_stats:
		player_stats[playerID]['G'] = player_stats[playerID]['G'] + row['G']
		player_stats[playerID]['AB'] = player_stats[playerID]['AB'] + row['AB']
		player_stats[playerID]['R'] = player_stats[playerID]['R'] + row['R']
		player_stats[playerID]['H'] = player_stats[playerID]['H'] + row['H']
		player_stats[playerID]['2B'] = player_stats[playerID]['2B'] + row['2B']
		player_stats[playerID]['3B'] = player_stats[playerID]['3B'] + row['3B']
		player_stats[playerID]['HR'] = player_stats[playerID]['HR'] + row['HR']
		player_stats[playerID]['RBI'] = player_stats[playerID]['RBI'] + row['RBI']
		player_stats[playerID]['SB'] = player_stats[playerID]['SB'] + row['SB']
		player_stats[playerID]['BB'] = player_stats[playerID]['BB'] + row['BB']
		player_stats[playerID]['SO'] = player_stats[playerID]['SO'] + row['SO']
		player_stats[playerID]['IBB'] = player_stats[playerID]['IBB'] + row['IBB']
		player_stats[playerID]['HBP'] = player_stats[playerID]['HBP'] + row['HBP']
		player_stats[playerID]['SH'] = player_stats[playerID]['SH'] + row['SH']
		player_stats[playerID]['SF'] = player_stats[playerID]['SF'] + row['SF']
		years_played[playerID].append(row['yearID'])
	else:
		player_stats[playerID] = {}
		player_stats[playerID]['G'] = row['G']
		player_stats[playerID]['AB'] = row['AB']
		player_stats[playerID]['R'] = row['R']
		player_stats[playerID]['H'] = row['H']
		player_stats[playerID]['2B'] = row['2B']
		player_stats[playerID]['3B'] = row['3B']
		player_stats[playerID]['HR'] = row['HR']
		player_stats[playerID]['RBI'] = row['RBI']
		player_stats[playerID]['SB'] = row['SB']
		player_stats[playerID]['BB'] = row['BB']
		player_stats[playerID]['SO'] = row['SO']
		player_stats[playerID]['IBB'] = row['IBB']
		player_stats[playerID]['HBP'] = row['HBP']
		player_stats[playerID]['SH'] = row['SH']
		player_stats[playerID]['SF'] = row['SF']
		years_played[playerID] = []
		years_played[playerID].append(row['yearID'])



		# Iterate through `years_played` and add the number of years played to `player_stats`
for k, v in years_played.items():
	player_stats[k]['Years_Played'] = len(list(set(v)))

	#time for fielding_df organizing
fielder_list=[]
for i, row in fielding_df.iterrows():
	playerID = row['playerID']
	Gf = row['G']
	GSf = row['GS']
	POf = row['PO']
	Af = row['A']
	Ef = row['E']
	DPf = row['DP']
	if playerID in player_stats and playerID in fielder_list:
		player_stats[playerID]['Gf'] = player_stats[playerID]['Gf'] + Gf
		player_stats[playerID]['GSf'] = player_stats[playerID]['GSf'] + GSf
		player_stats[playerID]['POf'] = player_stats[playerID]['POf'] + POf
		player_stats[playerID]['Af'] = player_stats[playerID]['Af'] + Af
		player_stats[playerID]['Ef'] = player_stats[playerID]['Ef'] + Ef
		player_stats[playerID]['DPf'] = player_stats[playerID]['DPf'] + DPf
	else:
		fielder_list.append(playerID)
		player_stats[playerID]['Gf'] = Gf
		player_stats[playerID]['GSf'] = GSf
		player_stats[playerID]['POf'] = POf
		player_stats[playerID]['Af'] = Af
		player_stats[playerID]['Ef'] = Ef
		player_stats[playerID]['DPf'] = DPf

	#time for awards_df organizing oooof

		#Dataframes for each award
mvp = awards_df[awards_df['awardID'] == 'Most Valuable Player']
roy = awards_df[awards_df['awardID'] == 'Rookie of the Year']
gg = awards_df[awards_df['awardID'] == 'Gold Glove']
ss = awards_df[awards_df['awardID'] == 'Silver Slugger']
ws_mvp = awards_df[awards_df['awardID'] == 'World Series MVP']

		# Include each DataFrame in `awards_list`
awards_list = [mvp, roy, gg, ss, ws_mvp]

		# Initialize lists for each of the above DataFrames
mvp_list = []
roy_list = []
gg_list = []
ss_list = []
ws_mvp_list = []
		
		# Include each of the above lists in `lists`
lists = [mvp_list,roy_list,gg_list,ss_list,ws_mvp_list] #lists[index] is yet another list

		# Add a count for each award for each player in `player_stats`
for index, v in enumerate(awards_list):
	for i, row in v.iterrows():
		playerID = row['playerID']
		award = row['awardID']
		if playerID in player_stats and playerID in lists[index]: # if the player's id is both in player stats and lists[index]
			player_stats[playerID][award] += 1
		else:
			lists[index].append(playerID)
			player_stats[playerID][award] = 1

	#organize allstar_df
allstar_list = []

for i,row in allstar_df.iterrows():
	playerID = row['playerID'] # put into the thing
	if playerID in player_stats and playerID in allstar_list: # if the player is already in the list, add an award
		player_stats[playerID]['AS_games'] += 1
	else: 
		allstar_list.append(playerID) #if not already in the list, add to list and start all star games counter
		player_stats[playerID]['AS_games'] = 1


	#organize hof_df
hof_df = hof_df[(hof_df['inducted'] == 'Y') & (hof_df['category'] == 'Player')] # filter `hof_df` to include only instances where a player was inducted into the Hall of Fame

for i, row in hof_df.iterrows(): # Indicate which players in `player_stats` were inducted into the Hall of Fame
	playerID = row['playerID']
	if playerID in player_stats:
		player_stats[playerID]['HoF'] = 1
		player_stats[playerID]['votedBy'] = row['votedBy']





#convert player_stats into a dataframe
stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
	# Add a column for playerID from the `stats_df` index
stats_df['playerID'] = stats_df.index
	#join stats_df and master_df
master_df = master_df.join(stats_df, on = 'playerID', how ='inner', rsuffix = 'mstr')




	#organize appearances.df
	#jk there's no organizing here




#need to collect info about what time period the player played in
pos_dict = {}
	# Iterate through `appearances_df`
	# Add a count for the number of appearances for each player at each position
	# Also add a count for the number of games played for each player in each era.
for i, row in appearances_df.iterrows():
	ID = row['playerID']
	year = row['yearID']
	if ID in pos_dict:
		pos_dict[ID]['G_all'] = pos_dict[ID]['G_all'] + row['G_all']
		pos_dict[ID]['G_p'] = pos_dict[ID]['G_p'] + row['G_p']
		pos_dict[ID]['G_c'] = pos_dict[ID]['G_c'] + row['G_c']
		pos_dict[ID]['G_1b'] = pos_dict[ID]['G_1b'] + row['G_1b']
		pos_dict[ID]['G_2b'] = pos_dict[ID]['G_2b'] + row['G_2b']
		pos_dict[ID]['G_3b'] = pos_dict[ID]['G_3b'] + row['G_3b']
		pos_dict[ID]['G_ss'] = pos_dict[ID]['G_ss'] + row['G_ss']
		pos_dict[ID]['G_lf'] = pos_dict[ID]['G_lf'] + row['G_lf']
		pos_dict[ID]['G_cf'] = pos_dict[ID]['G_cf'] + row['G_cf']
		pos_dict[ID]['G_rf'] = pos_dict[ID]['G_rf'] + row['G_rf']
		pos_dict[ID]['G_of'] = pos_dict[ID]['G_of'] + row['G_of']
		pos_dict[ID]['G_dh'] = pos_dict[ID]['G_dh'] + row['G_dh']
		if year < 1920:
			pos_dict[ID]['pre1920'] = pos_dict[ID]['pre1920'] + row['G_all']
		elif year >= 1920 and year <= 1941:
			pos_dict[ID]['1920-41'] = pos_dict[ID]['1920-41'] + row['G_all']
		elif year >= 1942 and year <= 1945:
			pos_dict[ID]['1942-45'] = pos_dict[ID]['1942-45'] + row['G_all']
		elif year >= 1946 and year <= 1962:
			pos_dict[ID]['1946-62'] = pos_dict[ID]['1946-62'] + row['G_all']
		elif year >= 1963 and year <= 1976:
			pos_dict[ID]['1963-76'] = pos_dict[ID]['1963-76'] + row['G_all']
		elif year >= 1977 and year <= 1992:
			pos_dict[ID]['1977-92'] = pos_dict[ID]['1977-92'] + row['G_all']
		elif year >= 1993 and year <= 2009:
			pos_dict[ID]['1993-2009'] = pos_dict[ID]['1993-2009'] + row['G_all']
		elif year > 2009:
			pos_dict[ID]['post2009'] = pos_dict[ID]['post2009'] + row['G_all']
	else:
		pos_dict[ID] = {}
		pos_dict[ID]['G_all'] = row['G_all']
		pos_dict[ID]['G_p'] = row['G_p']
		pos_dict[ID]['G_c'] = row['G_c']
		pos_dict[ID]['G_1b'] = row['G_1b']
		pos_dict[ID]['G_2b'] = row['G_2b']
		pos_dict[ID]['G_3b'] = row['G_3b']
		pos_dict[ID]['G_ss'] = row['G_ss']
		pos_dict[ID]['G_lf'] = row['G_lf']
		pos_dict[ID]['G_cf'] = row['G_cf']
		pos_dict[ID]['G_rf'] = row['G_rf']
		pos_dict[ID]['G_of'] = row['G_of']
		pos_dict[ID]['G_dh'] = row['G_dh']
		pos_dict[ID]['pre1920'] = 0
		pos_dict[ID]['1920-41'] = 0
		pos_dict[ID]['1942-45'] = 0
		pos_dict[ID]['1946-62'] = 0
		pos_dict[ID]['1963-76'] = 0
		pos_dict[ID]['1977-92'] = 0
		pos_dict[ID]['1993-2009'] = 0
		pos_dict[ID]['post2009'] = 0
		if year < 1920:
			pos_dict[ID]['pre1920'] = row['G_all']
		elif year >= 1920 and year <= 1941:
			pos_dict[ID]['1920-41'] = row['G_all']
		elif year >= 1942 and year <= 1945:
			pos_dict[ID]['1942-45'] = row['G_all']
		elif year >= 1946 and year <= 1962:
			pos_dict[ID]['1946-62'] = row['G_all']
		elif year >= 1963 and year <= 1976:
			pos_dict[ID]['1963-76'] = row['G_all']
		elif year >= 1977 and year <= 1992:
			pos_dict[ID]['1977-92'] = row['G_all']
		elif year >= 1993 and year <= 2009:
			pos_dict[ID]['1993-2009'] = row['G_all']
		elif year > 2009:
			pos_dict[ID]['post2009'] = row['G_all']

	# Convert the `pos_dict` to a DataFrame
pos_df = pd.DataFrame.from_dict(pos_dict, orient='index')




#determine the percentage of times each player played at each position
pos_col_list = pos_df.columns.tolist()

	#remove G_all string
pos_col_list.remove('G_all')

	#loop thru list, divide each column by the players total games played
for col in pos_col_list:
	column = col + '_percent'
	pos_df[column] = pos_df[col] / pos_df['G_all']

print(pos_df.head())#print out first rows of pos_df



#Filtering pos_df

pos_df = pos_df[(pos_df['G_p_percent'] < 0.1) & (pos_df['G_c_percent'] < 0.1)] #eliminate players who played 10% or more of their games as Pitchers or Catchers

# Join `pos_df` and `master_df`
master_df = master_df.join(pos_df,on='playerID',how='right')

#taking out ppl who didn't make it to HOF through voting

master_df['votedBy'] = master_df['votedBy'].fillna('None')# Replace NA values with `None`


master_df = master_df[(master_df['votedBy'] == 'None') | (master_df['votedBy'] == 'BBWAA') | (master_df['votedBy'] == 'Run Off')] # Filter `master_df` to include only players who were voted into the Hall of Fame or Players who did not make it at all



#converting bats/throws to a numeric
def bats_throws(col):
	if col == "R":
		return 1
	else:
		return 0

master_df['bats_R'] = master_df['bats'].apply(bats_throws)# Use the `apply()` method to create numeric columns from the bats and throws columns
master_df['throws_R'] = master_df['throws'].apply(bats_throws)



#parsing the date from the debute and finalgame
from datetime import datetime

master_df['debut'] =  pd.to_datetime(master_df['debut']) # Convert the `debut` column to datetime
master_df['finalGame'] = pd.to_datetime(master_df['finalGame']) # Convert the `finalGame` column to datetime

master_df['debutYear'] = pd.to_numeric(master_df['debut'].dt.strftime('%Y'), errors='coerce') # Create new columns for debutYear and finalYear
master_df['finalYear'] = pd.to_numeric(master_df['finalGame'].dt.strftime('%Y'), errors='coerce')



#dropping unnecessary values
df = master_df.drop(['votedBy', 'IBB', 'bats', 'throws', 'GSf', 'POf','Gf', 'playerIDmstr'], axis=1)

print(df.columns)# Print `df` columns 

print(df.isnull().sum(axis=0).tolist())# Print a list of null values



# Fill null values in numeric columns with 0
fill_cols = ['AS_games', 'Silver Slugger', 'Rookie of the Year', 'Gold Glove', 'Most Valuable Player', 'HoF', '1977-92', 'pre1920', '1942-45', '1946-62', '1963-76', '1920-41', '1993-2009', 'HBP', 'SB', 'SF', 'SH', 'RBI', 'SO', 'World Series MVP', 'G_dh_percent', 'G_dh', 'Af', 'DPf', 'Ef']

for col in fill_cols:
    df[col] = df[col].fillna(0)

df = df.dropna()# Drop any additional rows with null values

print(df.isnull().sum(axis=0).tolist())# Check to make sure null values have been removed



###########################################FEATURE ENGINEERING#############################################


# Create Batting Average (`AVE`) column
df['AVE'] = df['H'] / df['AB']

# Create On Base Percent (`OBP`) column
plate_appearances = (df['AB'] + df['BB'] + df['SF'] + df['SH'] + df['HBP'])
df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / plate_appearances

# Create Slugging Percent (`Slug_Percent`) column
single = ((df['H'] - df['2B']) - df['3B']) - df['HR']
df['Slug_Percent'] = ((df['HR'] * 4) + (df['3B'] * 3) + (df['2B'] * 2) + single) / df['AB']

# Create On Base plus Slugging Percent (`OPS`) column
hr = df['HR'] * 4
triple = df['3B'] * 3
double = df['2B'] * 2
df['OPS'] = df['OBP'] + df['Slug_Percent']


# Eliminate Jackson and Rose from `df`
df = df[(df['playerID'] != 'rosepe01') & (df['playerID'] != 'jacksjo01')]


# Make a function that will create a new column to honor Jackie Robinson, the first African American Major league Baseball Player
def first_aap(col):
    if col == 'robinja02':
        return 1
    else:
        return 0
    
# Apply `first_aap` to `df['playerID']`    
df['first_aap'] = df['playerID'].apply(first_aap)


# Filter the `df` for the remaining Hall of Fame members in the data
df_hof = df[df['HoF'] == 1]



#PLOTTING WOOT
# Import the `pyplot` module from `matplotlib`
import matplotlib.pyplot as plt

# Initialize the figure and add subplots
fig = plt.figure(figsize=(15, 12))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)


#HISTOGRAM CODE
# Create distribution plots for Hits, Home Runs, Years Played and All Star Games
ax1.hist(df_hof['H'])
ax1.set_title('Distribution of Hits')
ax1.set_ylabel('HoF Careers')
ax2.hist(df_hof['HR'])
ax2.set_title('Distribution of Home Runs')
ax3.hist(df_hof['Years_Played'])
ax3.set_title('Distribution of Years Played')
ax3.set_ylabel('HoF Careers')
ax4.hist(df_hof['AS_games'])
ax4.set_title('Distribution of All Star Game Appearances')

# Show the plot
plt.show()


#SCATTER PLOT
# Filter `df` for players with 10 or more years of experience
df_10 = df[(df['Years_Played'] >= 10) & (df['HoF'] == 0)]

# Initialize the figure and add subplots
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# Create Scatter plots for Hits vs. Average and Home Runs vs. Average
ax1.scatter(df_hof['H'], df_hof['AVE'], c='r', label='HoF Player')
ax1.scatter(df_10['H'], df_10['AVE'], c='b', label='Non HoF Player')
ax1.set_title('Career Hits vs. Career Batting Average')
ax1.set_xlabel('Career Hits')
ax1.set_ylabel('Career Average')
ax2.scatter(df_hof['HR'], df_hof['AVE'], c='r', label='HoF Player')
ax2.scatter(df_10['HR'], df_10['AVE'], c='b', label='Non HoF Player')
ax2.set_title('Career Home Runs vs. Career Batting Average')
ax2.set_xlabel('Career Home Runs')
ax2.legend(loc='lower right', scatterpoints=1)

# Show the plot
plt.show()



# Check for null values
print(df.isnull().sum(axis=0).tolist())

# Eliminate rows with null values
df = df.dropna()



# Create column for years since retirement
df['YSLS'] = 2016 - df['finalYear']

# Filter `df` for players who retired more than 15 years ago
df_hitters = df[df['YSLS'] > 15]

print(df_hitters.head())

# Filter `df` for players who retired less than 15 years ago and for currently active players
df_eligible = df[df['YSLS']<= 15]

print(df_eligible.head())



# Select columns to use for models, and identification columns
num_cols_hitters = ['playerID', 'nameFirst', 'nameLast', 'HoF', 'Years_Played', 'H', 'BB', 'HR', 'AVE', 'OBP', 'Slug_Percent', 'OPS',  'RBI','R', 'SB', '2B', '3B', 'AB', 'SO', 'Most Valuable Player', 'World Series MVP', 'AS_games','Gold Glove', 'Rookie of the Year', 'Silver Slugger', 'bats_R', 'throws_R', 'DPf', 'Af', 'Ef', 'YSLS', 'G_all', '1963-76_percent', '1993-2009_percent', '1946-62_percent', 'G_1b_percent', '1942-45_percent','G_dh_percent', '1920-41_percent', 'G_ss_percent','post2009_percent', '1977-92_percent', 'G_2b_percent', 'G_3b_percent','G_of_percent', 'pre1920_percent', 'first_aap']

# Create a new DataFrame (`data`) from the `df_hitters` using the columns above
data = df_hitters[num_cols_hitters]

# Return the first rows of `data`
print(data.head())



# Create `target` Series
target = data['HoF']

# Create `features` DataFrame
features = data.drop(['playerID', 'nameFirst', 'nameLast', 'HoF'], axis=1)



#LOGISTIC REGRESSION TIME
# Import cross_val_predict, KFold and LogisticRegression from 'sklearn'
from sklearn.cross_validation import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
lr = LogisticRegression(class_weight='balanced')

# Create an instance of the KFold class
kf = KFold(features.shape[0], random_state=1)

# Create predictions using cross validation
predictions_lr = cross_val_predict(lr, features, target, cv=kf)




# Import NumPy as np
import numpy as np

# Convert predictions and target to NumPy arrays
np_predictions_lr = np.asarray(predictions_lr)
np_target = target.as_matrix()


# Determine True Positive count
tp_filter_lr = (np_predictions_lr == 1) & (np_target == 1)
tp_lr = len(np_predictions_lr[tp_filter_lr])

# Determine False Negative count
fn_filter_lr = (np_predictions_lr == 0) & (np_target == 1)
fn_lr = len(np_predictions_lr[fn_filter_lr])

# Determine False Positive count
fp_filter_lr = (np_predictions_lr == 1) & (np_target == 0)
fp_lr = len(np_predictions_lr[fp_filter_lr])

# Determine True Negative count
tn_filter_lr = (np_predictions_lr == 0) & (np_target == 0)
tn_lr = len(np_predictions_lr[tn_filter_lr])

# Determine True Positive rate
tpr_lr = tp_lr / (tp_lr + fn_lr)

# Determine False Negative rate
fnr_lr = fn_lr / (fn_lr + tp_lr)

# Determine False Positive rate
fpr_lr = fp_lr / (fp_lr + tn_lr)

# Print each count
print(tp_lr)
print(fn_lr)
print(fp_lr)

# Print each rate
print(tpr_lr)
print(fnr_lr)
print(fpr_lr)


# Import RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# Create penalty dictionary
penalty = {
    0: 100,
    1: 1
}



#RANDOM FOREST MODEL WOOT
# Create Random Forest model
rf = RandomForestClassifier(random_state=1,n_estimators=12, max_depth=11, min_samples_leaf=1, class_weight=penalty)

# Create predictions using cross validation
predictions_rf = cross_val_predict(rf, features, target, cv=kf)

# Convert predictions to NumPy array
np_predictions_rf = np.asarray(predictions_rf)

# Determine True Positive count
tp_filter_rf = (np_predictions_rf == 1) & (np_target == 1)
tp_rf = len(np_predictions_rf[tp_filter_rf])

# Determine False Negative count
fn_filter_rf = (np_predictions_rf == 0) & (np_target == 1)
fn_rf = len(np_predictions_rf[fn_filter_rf])

# Determine False Positive count
fp_filter_rf = (np_predictions_rf == 1) & (np_target == 0)
fp_rf = len(np_predictions_rf[fp_filter_rf])

# Determine True Negative count
tn_filter_rf = (np_predictions_rf == 0) & (np_target == 0)
tn_rf = len(np_predictions_rf[tn_filter_rf])

# Determine True Positive rate
tpr_rf = tp_rf / (tp_rf + fn_rf)

# Determine False Negative rate
fnr_rf = fn_rf / (fn_rf + tp_rf)

# Determine False Positive rate
fpr_rf = fp_rf / (fp_rf + tn_rf)

# Print each count
print(tp_rf)
print(fn_rf)
print(fp_rf)

# Print each rate
print(tpr_rf)
print(fnr_rf)
print(fpr_rf)


# Create a new DataFrame from `df_eligible` using `num_col_hitters`
new_data = df_eligible[num_cols_hitters]

# Create a new features DataFrame
new_features = new_data.drop(['playerID', 'nameFirst', 'nameLast', 'HoF'], axis=1)


# Fit the Random Forest model
rf.fit(features, target)

# Estimate probabilities of Hall of Fame induction
probabilities = rf.predict_proba(new_features)



# Convert predictions to a DataFrame
hof_predictions = pd.DataFrame(probabilities[:,1])

# Sort the DataFrame (descending)
hof_predictions = hof_predictions.sort_values(0, ascending=False)

hof_predictions['Probability'] = hof_predictions[0]

# Print 50 highest probability HoF inductees from still eligible players
for i, row in hof_predictions.head(50).iterrows():
    prob = ' '.join(('HoF Probability =', str(row['Probability'])))
    print('')
    print(prob)
    print(new_data.iloc[i,1:27])








