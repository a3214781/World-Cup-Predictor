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

# 2. THE CHROME-PLATED FORM EXTRACTOR FUNCTION
def get_team_form(df, current_idx, team, num_games=5):
    """Looks backward from current_idx to find the last 5 game outcomes for a team"""
    # Grab all historical rows BEFORE this current game
    past_games = df.iloc[:current_idx]
    
    # Filter for games where our target team actually played
    team_history = past_games[(past_games['home_team'] == team) | (past_games['away_team'] == team)]
    
    # Get the last N games
    last_games = team_history.tail(num_games)
    
    # Map out Wins (W) and Losses (L)
    form_list = []
    for _, match in last_games.iterrows():
        if match['winner'] == team:
            form_list.append('W')
        else:
            form_list.append('L') # No draws exist in your dataset anymore!
            
    # Join into a string like "WWLLW". Pad with 'N' if they haven't played 5 games yet
    form_str = "".join(form_list)
    return form_str.rjust(num_games, 'N') 


# 3. GENERATE FORM FEATURES FOR THE WHOLE DATASET
home_forms = []
away_forms = []

print("⚡ Calculating team forms... Hang tight!")
for idx, row in data.iterrows():
    home_forms.append(get_team_form(data, idx, row['home_team']))
    away_forms.append(get_team_form(data, idx, row['away_team']))

# Assign them as new columns to feed into your binary regression!
data['home_form'] = home_forms
data['away_form'] = away_forms


# 4. PRINT A SNEAK PEEK OF YOUR BRAND NEW FEATURES
print("\n🔥 SUCCESS! Check out your new model features:")
print(data[['date', 'home_team', 'away_team', 'winner', 'home_form', 'away_form']].tail(10))

# What Info I Have
# Form/ Team Coming into the game
# Result (17 games into world cup)
# 100% Accuracy (17/17) - Pure Binary
# 68.0% Accuracy (17/25) - Real World Realism Mode (Cant predict draws)
# python3 prediction-v2.py