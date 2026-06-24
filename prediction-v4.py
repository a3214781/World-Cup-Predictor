import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Gathering results.csv as a variable data
data = pd.read_csv("/Users/alessiofantasia/prediction-wc/data/results.csv", parse_dates=[0])
rankings = pd.read_csv('/Users/alessiofantasia/prediction-wc/data/fifa_ranking.csv', parse_dates=['rank_date']).rename(columns={'rank_date': 'date'})

# Sorting data by dat adn rank before merging
data = data.sort_values('date')
rankings = rankings.sort_values('date')

# filter out all friendly matches and matches predating 2010
data = data[data['tournament'] != 'Friendly']
data = data.dropna(subset=['home_score', 'away_score'])
data = data[data['date'] >= '2010-01-01']

# Merging datasets together by hometeam and awaytem
home_rankings = pd.merge_asof(data, rankings[['date', 'country_full', 'rank']], on='date', left_by='home_team', right_by='country_full').rename(columns={'rank': 'home_rank'})
away_rankings = pd.merge_asof(data, rankings[['date', 'country_full', 'rank']], on='date', left_by='away_team', right_by='country_full').rename(columns={'rank': 'away_rank'})

# Create table merging both home and away ranking tables together
final_rankings = pd.merge(home_rankings, away_rankings, on=['date', 'home_team', 'away_team'])

# Conditions for winning and loosing
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

# Create 'winner' column using np.select
winner = np.select(conditions, choices)
data['winner'] = winner

# Dropping all draws from datased for binary regression (win or lose)
data = data[data['winner'] != 'Draw']

# Create a row for each team's participation in each match
home = data[['date', 'home_team', 'winner', 'home_score', 'away_score']].copy()
home.columns = ['date', 'home_team', 'winner', 'home_score', 'away_score']
home['win'] = (home['winner'] == home['home_team']).astype(int)
home = home.rename(columns={'home_team': 'team'})
home['goals_scored'] = home['home_score']
home['goals_conceded'] = home['away_score']

# Same for away teams
away = data[['date', 'away_team', 'winner', 'home_score', 'away_score']].copy()
away.columns = ['date', 'away_team', 'winner', 'home_score', 'away_score']
away['win'] = (away['winner'] == away['away_team']).astype(int)
away = away.rename(columns={'away_team': 'team'})
away['goals_scored'] = away['away_score']
away['goals_conceded'] = away['home_score']


# Combine and sort by date
team_results = pd.concat([home, away]).sort_values('date').reset_index(drop=True)

team_results['form'] = team_results.groupby('team')['win'].rolling(5).mean().shift(1).values
team_results['avg_goals_scored'] = team_results.groupby('team')['goals_scored'].rolling(5).sum().shift(1).values
team_results['avg_goals_conceded'] = team_results.groupby('team')['goals_conceded'].rolling(5).sum().shift(1).values

# Merge home team form and goal stats onto main dataframe
home_team_form = pd.merge(data, team_results, left_on=['date', 'home_team'], right_on=['date', 'team']).rename(columns={
    'form': 'home_team_form',
    'avg_goals_scored': 'home_avg_goals_scored',
    'avg_goals_conceded': 'home_avg_goals_conceded'
})

# Same for away team
away_team_form = pd.merge(data, team_results, left_on=['date', 'away_team'], right_on=['date', 'team']).rename(columns={
    'form': 'away_team_form',
    'avg_goals_scored': 'away_avg_goals_scored',
    'avg_goals_conceded': 'away_avg_goals_conceded'
})

# New dataframe with date, home_team, away_team, winner merging both forms
data_final = pd.merge(home_team_form, away_team_form, on=['date', 'home_team', 'away_team', 'winner_x'])
data_final = pd.merge(data_final, final_rankings, on=['date', 'home_team', 'away_team'])
data_final = pd.merge(data_final, data[['date', 'home_team', 'away_team', 'neutral']], on=['date', 'home_team', 'away_team'], how='left')

# Drop Null Values
data_final = data_final.dropna(subset=['home_team_form', 'away_team_form', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded', 'home_rank', 'away_rank'])

data_final['home_rank'] = data_final['home_rank']
data_final['away_rank'] = data_final['away_rank']

# finds wich teams won and converts boolean to binary
data_final['target'] = (data_final['winner_x'] == data_final['home_team']).astype(int)

# defining X and Y for the Binomial Regression
X = data_final[['home_team_form', 'away_team_form', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded', 'home_rank', 'away_rank', 'neutral']]
Y = data_final['target']

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(
  X,Y ,random_state=42, test_size=0.25, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)


# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

'''
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

'''

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred),"%")

def predict_match(home_team, away_team, neutral=False):
    
    # Cleaning data before it gets analysed 
    team_results_clean = team_results.dropna(subset=['form', 'avg_goals_scored', 'avg_goals_conceded'])

    # Finding current teams most recent rank
    home_rank = rankings[rankings['country_full'] == home_team].iloc[-1]['rank']
    away_rank = rankings[rankings['country_full'] == away_team].iloc[-1]['rank']
    
    # Finding current teams 5 game running form
    home_form = team_results_clean[team_results_clean['team'] == home_team].iloc[-1]['form']
    away_form = team_results_clean[team_results_clean['team'] == away_team].iloc[-1]['form']
    
    # Finding goals conceded and scored off recent form
    home_avg_conceded = team_results_clean[team_results_clean['team'] == home_team].iloc[-1]['avg_goals_conceded']
    home_avg_scored = team_results_clean[team_results_clean['team'] == home_team].iloc[-1]['avg_goals_scored']
    
    away_avg_conceded = team_results_clean[team_results_clean['team'] == away_team].iloc[-1]['avg_goals_conceded']
    away_avg_scored = team_results_clean[team_results_clean['team'] == away_team].iloc[-1]['avg_goals_scored']
    
    # Create a dataframe with all relevant statistices to help predict
    match = pd.DataFrame([[home_form, away_form, home_avg_scored, away_avg_scored, home_avg_conceded, away_avg_conceded, home_rank, away_rank, int(neutral)]], 
        columns=['home_team_form', 'away_team_form', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded', 'home_rank', 'away_rank', 'neutral'])
    
    # Call logreg.predict_proba(match) to predict the match
    match_scaled = scaler.transform(match)
    probabilities = logreg.predict_proba(match_scaled)
    
    # print(home_team, ' win probability: ', round(probabilities[0][1] * 100, 1), '%')
    # print(away_team, ' win probability: ', round(probabilities[0][0] * 100, 1), '%')
    
    home_probability = probabilities[0][1] * 100
    away_probability = probabilities[0][0] * 100
    
    return home_probability, away_probability

# Array of all groups at the 2026 WC
wc_groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Generate all 6 matches per group (round robin - each team plays every other team once)
wc_matches = []
# Loop through all groups and generate every unique pair of teams
for group, teams in wc_groups.items():
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            # Append (home, away, neutral, group) for each matchup
            wc_matches.append((teams[i], teams[j], True, group))

# Print and predict every group stage match
     
# Dictionary to track points for every team, starting at 0
points = {team: 0 for group in wc_groups.values() for team in group}
total_probability = {team: 0 for group in wc_groups.values() for team in group}

# Loop through all 72 matches and assign points based on predicted winner
for home, away, neutral, group in wc_matches:
    try:
        # Get win probabilities for each team
        home_prob, away_prob = predict_match(home, away, neutral)
        # Adding total_probibility to each home and away team
        total_probability[home] += home_prob
        total_probability[away] += away_prob
        # Award 3 points to the predicted winner
        if home_prob > away_prob:
            points[home] += 3
        else:
            points[away] += 3
    except Exception as e:
        print(f"Error: {home} vs {away} — {e}")
        
# Group X: Home vs Away
# Home win probability:  X%
# Away win probability:  Y%

for home, away, neutral, group in wc_matches:
    # print(f"Group {group}: {home} vs {away}")
    try:
        predict_match(home, away, neutral)
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()

# Function used to return the points tally for each team (used to sort teams in their group)        
def get_points(team):
    return points[team]

def get_prob_total(team):
    return total_probability[team]
# Print final group tables

third_place = []
for group, teams in wc_groups.items():
    print(f"\n--- Group {group} ---")
    # Sort teams in this group by their points (highest first)
    sorted_teams = sorted(teams, key=get_points, reverse=True)
    for i, team in enumerate(sorted_teams):
        print(f"{i+1}. {team} — {points[team]} pts")
        # add all 3rd place teams to a list to find top 8 to go through to the knockout stage
    third_place.append(sorted_teams[2])
    
third_place = sorted(third_place, key=get_prob_total, reverse=True)

print(third_place)
'''
while True:
    home_team_input = input("Enter home team (or Q to quit): ")
    if home_team_input.upper() == 'Q':
        break
    away_team_input = input("Enter away team: ")
    neutral_input = input("Neutral ground? (y/n): ").lower() == 'y'
    predict_match(home_team_input, away_team_input, neutral_input)
    print()
    
'''
# python3 prediction-v4.py
