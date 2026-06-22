# KGAT_b4bb4a3970b94af8d610d313c5b047aa
# python3 prediction-v2.py

import pandas as pd
import numpy as np

data = pd.read_csv("/Users/alessiofantasia/prediction-wc/data/results.csv", parse_dates=[0])

# filter out all friendly matches
data = data[data['tournament'] != 'Friendly']
data = data.dropna(subset=['home_score', 'away_score'])
data = data[data['date'] >= '2010-01-01']


# np.where(condition, value_if_true, value_if_false)

conditions = [
    data['home_score'] > data['away_score'],  # Home win
    data['away_score'] > data['home_score'],  # Away win
    data['home_score'] == data['away_score']  # Draw
]

choices = [
    data['home_team'],  # Result if condition 1 is True
    data['away_team'],  # Result if condition 2 is True
    'Draw'              # Result if condition 3 is True
]

# Create 'winner' column
winner = np.select(conditions, choices)
data['winner'] = winner

# Dropping all draws from datased for binary regression (win or lose)
data = data[data['winner'] != 'Draw']

def form(data, match_date, team, num_games=5):
    # Filters data for specific team
    data = data[(data['home_team'] == team) | (data['away_team'] == team)]
    # Filters to only show games that happened BEFORE the current match date
    data = data[data['date'] < match_date]
    
    data = data.tail(num_games)
    
    win_rate = (data['winner'] == team).sum() / len(data)
    
    return win_rate
    
    
print(form(data, '2022-01-01', 'Australia', num_games=5))

# What Info I Have
# Form/ Team Coming into the game
# Result (17 games into world cup)
# 100% Accuracy (18/18) - Pure Binary
# Accuracy (18/26) - Real World Realism Mode (Cant predict draws)
# python3 prediction-v2.py