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

print(data['winner'].value_counts()) 


# python3 prediction-v2.py
