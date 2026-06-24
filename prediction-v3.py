import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Gathering results.csv as a variable data
data = pd.read_csv("/Users/alessiofantasia/prediction-wc/data/results.csv", parse_dates=[0])

# filter out all friendly matches
data = data[data['tournament'] != 'Friendly']
data = data.dropna(subset=['home_score', 'away_score'])
data = data[data['date'] >= '2010-01-01']

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
data_final = data_final.dropna(subset=['home_team_form', 'away_team_form', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded'])

# finds wich teams won and converts boolean to binary
data_final['target'] = (data_final['winner_x'] == data_final['home_team']).astype(int)

print(data_final['target'].value_counts())

# defining X and Y for the Binomial Regression
X = data_final[['home_team_form', 'away_team_form', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded']]
Y = data_final['target']

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(
  X,Y ,random_state=42, test_size=0.25, shuffle=True)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

print(metrics.accuracy_score(y_test, y_pred))


# print(len(data))
# print(data[['home_team', 'away_team', 'home_team_form', 'away_team_form']].head(20))

  
# python3 prediction-v3.py
