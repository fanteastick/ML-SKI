import pandas as pd

#importing all the data from CSV files
master_df = pd.read_csv('People.csv', usecols=['playerID', 'nameFirst', 'nameLast', 'bats', 'throws', 'debut', 'finalGame'])
fielding_df = pd.read_csv('Fielding.csv',usecols=['playerID','yearID','stint','teamID','lgID','POS','G','GS','InnOuts','PO','A','E','DP'])
batting_df = pd.read_csv('Batting.csv')
awards_df = pd.read_csv('AwardsPlayers.csv', usecols=['playerID','awardID','yearID'])
allstar_df = pd.read_csv('AllstarFull.csv', usecols=['playerID','yearID'])
hof_df = pd.read_csv('HallOfFame.csv',usecols=['playerID','yearid','votedBy','needed_note','inducted','category'])
appearances_df = pd.read_csv('Appearances.csv')




#data cleaning and preprocessing

	#start w/ batting_df
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
			player_stats[k]['Years_Played']=len(list(set(v)))




