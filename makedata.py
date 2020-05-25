import pandas as pd
import numpy as np
import random as rand

# stats = ['ADJOE','ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB','DRB', 'FTR', 'FTRD', '2P_O','2P_D','3P_O','3P_D', 'ADJ_T','WAB','SEED']

def make_data(year, stats):
    
    cbb15 = pd.read_csv(f'datasets/cbb{year}.csv')
    madness_results = pd.read_csv('datasets/ncaa1516results.csv')
    march_team_data = cbb15.loc[cbb15['SEED'] > 0]
    matches = madness_results.loc[1916:1982][['Winner', 'Loser', 'Winning Score', 'Losing Score']]

    converts = {"St. John's, New York" : "St. John's",
                'California-Irvine' : 'UC Irvine',
                'Virginia Commonwealth' : 'VCU',
                'Southern Methodist' : 'SMU',
                'Louisiana St.' : 'LSU',
                'Brigham Young' : 'BYU',
                'Xavier, Ohio' : 'Xavier'}

    train_data = []
    label_data = []
    pairs = []

    for idx, (team1, team2) in enumerate(zip(matches['Winner'].tolist(), matches['Loser'].tolist())):
        team1 = team1.replace('State', 'St.')
        team2 = team2.replace('State', 'St.')
        team1 = converts[team1] if team1 in converts.keys() else team1
        team2 = converts[team2] if team2 in converts.keys() else team2

        team1_data = march_team_data.loc[march_team_data['TEAM'] == team1][stats]
        team2_data = march_team_data.loc[march_team_data['TEAM'] == team2][stats]

        point_diff = matches.iloc[idx, 2] - matches.iloc[idx, 3]

        win = rand.choice([True, False])

        if win:
            if point_diff >= 15:
                label_data.append(0)
            elif point_diff >= 1:
                label_data.append(1)
            # train_data.append((team1, team2, team1_data.values[0] - team2_data.values[0]))
            train_data.append(team1_data.values[0] - team2_data.values[0])
            pairs.append((team1, team2))
        else:
            if point_diff >= 15:
                label_data.append(2)
            elif point_diff >= 1:
                label_data.append(3)
            # train_data.append((team2, team1, team2_data.values[0] - team1_data.values[0]))
            train_data.append(team2_data.values[0] - team1_data.values[0])
            pairs.append((team2, team1))

    return (train_data, label_data, pairs)





