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
    "A": ["🇲🇽 Mexico", "🇿🇦 South Africa", "🇰🇷 South Korea", "🇨🇿 Czechia"],
    "B": ["🇨🇦 Canada", "🇨🇭 Switzerland", "🇶🇦 Qatar", "🇧🇦 Bosnia and Herzegovina"],
    "C": ["🇧🇷 Brazil", "🇲🇦 Morocco", "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland", "🇭🇹 Haiti"],
    "D": ["🇺🇸 United States", "🇵🇾 Paraguay", "🇦🇺 Australia", "🇹🇷 Turkey"],
    "E": ["🇩🇪 Germany", "🇨🇼 Curaçao", "🇨🇮 Ivory Coast", "🇪🇨 Ecuador"],
    "F": ["🇳🇱 Netherlands", "🇯🇵 Japan", "🇹🇳 Tunisia", "🇸🇪 Sweden"],
    "G": ["🇧🇪 Belgium", "🇪🇬 Egypt", "🇮🇷 Iran", "🇳🇿 New Zealand"],
    "H": ["🇪🇸 Spain", "🇨🇻 Cape Verde", "🇸🇦 Saudi Arabia", "🇺🇾 Uruguay"],
    "I": ["🇫🇷 France", "🇸🇳 Senegal", "🇳🇴 Norway", "🇮🇶 Iraq"],
    "J": ["🇦🇷 Argentina", "🇩🇿 Algeria", "🇦🇹 Austria", "🇯🇴 Jordan"],
    "K": ["🇵🇹 Portugal", "🇨🇴 Colombia", "🇺🇿 Uzbekistan", "🇨🇩 DR Congo"],
    "L": ["🏴󠁧󠁢󠁥󠁮󠁧󠁿 England", "🇭🇷 Croatia", "🇬🇭 Ghana", "🇵🇦 Panama"],
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
        # print()
    except Exception as e:
        print(f"Error: {e}")
        # print()

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

print("\n--- Top 8 Third Place Teams ---")
for i, team in enumerate(third_place[:8]):
    print(f"{i+1}. {team} — {round(total_probability[team], 1)}% total probability")
    
# ─────────────────────────────────────────────────────────
# ROUND OF 32 BRACKET
# Based on FIFA combination 259 (3rd place teams from groups A,C,D,E,G,I,J,K)
# Note: this is hardcoded for our specific predicted 3rd place outcome
# In reality the combination depends on which groups produce 3rd place qualifiers
# ─────────────────────────────────────────────────────────

# Helps to get group winner and runner up and third place from sorted groups
def get_winner(group):
    teams = sorted(wc_groups[group], key=get_points, reverse=True)
    return teams[0]

def get_runner_up(group):
    teams = sorted(wc_groups[group], key=get_points, reverse=True)
    return teams[1]

def get_third(group):
    teams = sorted(wc_groups[group], key=get_points, reverse=True)
    return teams[2]

# 16 Round of 32 matchups
round_of_32 = [
    # Fixed matchups (runner up vs runner up)
    (get_runner_up("A"), get_runner_up("B")),   # South Korea vs Canada
    (get_winner("C"),    get_runner_up("F")),    # Brazil vs Japan
    (get_winner("F"),    get_runner_up("C")),    # Netherlands vs Morocco
    (get_runner_up("E"), get_runner_up("I")),    # Ecuador vs Senegal
    (get_runner_up("K"), get_runner_up("L")),    # Colombia vs Croatia
    (get_winner("H"),    get_runner_up("J")),    # Spain vs Algeria
    (get_winner("J"),    get_runner_up("H")),    # Argentina vs Uruguay
    (get_runner_up("D"), get_runner_up("G")),    # Australia vs Iran
    
    # Group winner vs 3rd place (combination 259)
    (get_winner("A"),    get_third("E")),        # Mexico vs Ivory Coast
    (get_winner("B"),    get_third("G")),        # Switzerland vs Egypt
    (get_winner("D"),    get_third("J")),        # USA vs Austria
    (get_winner("E"),    get_third("C")),        # Germany vs Scotland
    (get_winner("G"),    get_third("A")),        # Belgium vs Czechia
    (get_winner("I"),    get_third("D")),        # France vs Paraguay
    (get_winner("K"),    get_third("I")),        # Portugal vs Norway
    (get_winner("L"),    get_third("K")),        # England vs Uzbekistan
]

# Print and simulate the Round of 32
print("\n========== ROUND OF 32 ==========")
round_of_32_winners = []
for home, away in round_of_32:
    print(f"\n{home} vs {away}")
    try:
        home_prob, away_prob = predict_match(home, away, True)
        print(f"{home}: {round(home_prob, 1)}%  |  {away}: {round(away_prob, 1)}%")
        winner = home if home_prob > away_prob else away
        print(f"→ Winner: {winner}")
        round_of_32_winners.append(winner)
    except Exception as e:
        print(f"Error: {e}")

# list to string to pair teams before the next round
round_of_16 = [(round_of_32_winners[i], round_of_32_winners[i+1]) 
               for i in range(0, len(round_of_32_winners), 2)]

# Print and simulate the Round of 16
print("\n========== ROUND OF 16 ==========")
round_of_16_winners = []
for home, away in round_of_16:
    print(f"\n{home} vs {away}")
    try:
        home_prob, away_prob = predict_match(home, away, True)
        print(f"{home}: {round(home_prob, 1)}%  |  {away}: {round(away_prob, 1)}%")
        winner = home if home_prob > away_prob else away
        print(f"→ Winner: {winner}")
        round_of_16_winners.append(winner)
    except Exception as e:
        print(f"Error: {e}")

# list to string to pair teams before the next round
quarter_finalists = [(round_of_16_winners[i], round_of_16_winners[i+1]) 
               for i in range(0, len(round_of_16_winners), 2)]

# Print and simulate the Quarter Finals
print("\n========== Quarter Finals ==========")
quarter_final_winners = []
for home, away in quarter_finalists:
    print(f"\n{home} vs {away}")
    try:
        home_prob, away_prob = predict_match(home, away, True)
        print(f"{home}: {round(home_prob, 1)}%  |  {away}: {round(away_prob, 1)}%")
        winner = home if home_prob > away_prob else away
        print(f"→ Winner: {winner}")
        quarter_final_winners.append(winner)
    except Exception as e:
        print(f"Error: {e}")

# list to string to pair teams before the next round
semi_finalists = [(quarter_final_winners[i], quarter_final_winners[i+1]) 
               for i in range(0, len(quarter_final_winners), 2)]
        
# Print and simulate the Semi Finals
print("\n========== Semi Finals ==========")
semi_finalists_winners = []
for home, away in semi_finalists:
    print(f"\n{home} vs {away}")
    try:
        home_prob, away_prob = predict_match(home, away, True)
        print(f"{home}: {round(home_prob, 1)}%  |  {away}: {round(away_prob, 1)}%")
        winner = home if home_prob > away_prob else away
        print(f"→ Winner: {winner}")
        semi_finalists_winners.append(winner)
    except Exception as e:
        print(f"Error: {e}")
        
# list to string to pair teams before the next round
grand_finalist = [(semi_finalists_winners[i], semi_finalists_winners[i+1]) 
               for i in range(0, len(semi_finalists_winners), 2)]
        
# Print and simulate the World Cup Final
print("\n========== World Cup Final ==========")
grand_finalist_winners = []
for home, away in grand_finalist:
    print(f"\n{home} vs {away}")
    try:
        home_prob, away_prob = predict_match(home, away, True)
        print(f"{home}: {round(home_prob, 1)}%  |  {away}: {round(away_prob, 1)}%")
        winner = home if home_prob > away_prob else away
        print(f" The 2026 World Cup Winner Is: {winner}")
        grand_finalist_winners.append(winner)
    except Exception as e:
        print(f"Error: {e}")

# python3 prediction-v4.py


'''
# Generate HTML report
# Generate professional HTML report
winner = grand_finalist_winners[0] if grand_finalist_winners else "TBD"

html = """<!DOCTYPE html>
<html>
<head>
<title>World Cup 2026 AI Simulation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
* { box-sizing: border-box; }
body { margin: 0; background: radial-gradient(circle at top, #17324d, #05070b 55%); color: #f5f5f5; font-family: Inter, Arial; }
.container { max-width: 1200px; margin: auto; padding: 40px 25px; }
.hero { text-align: center; padding: 70px 20px; }
.hero h1 { font-size: 60px; margin: 0; font-weight: 900; background: linear-gradient(90deg, #ffffff, #9bdcff); -webkit-background-clip: text; color: transparent; }
.hero p { color: #9ca3af; font-size: 18px; }
.section { margin-top: 60px; }
.section-title { font-size: 30px; font-weight: 800; margin-bottom: 25px; }
.groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
.group { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(15px); border-radius: 20px; padding: 22px; transition: .3s; }
.group:hover { transform: translateY(-5px); }
.group h3 { margin-top: 0; color: #7dd3fc; }
.team { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,.08); color: #ddd; }
.rank1 { color: #ffd700; font-weight: 800; }
.rank2 { color: #60a5fa; }
.matches { display: flex; flex-direction: column; gap: 14px; }
.match { background: rgba(255,255,255,.05); padding: 18px 22px; border-radius: 15px; display: flex; justify-content: space-between; align-items: center; }
.win { color: #4ade80; font-weight: 700; }
.round-title { margin-top: 35px; font-size: 22px; color: #93c5fd; }
.final { margin-top: 70px; padding: 50px; text-align: center; background: linear-gradient(135deg, #172554, #0f172a); border-radius: 30px; border: 1px solid #2563eb; }
.final h2 { font-size: 45px; margin: 10px; }
.trophy { font-size: 70px; }
</style>
</head>
<body>
<div class="container">
<div class="hero">
<h1>🏆 World Cup 2026 AI Simulation</h1>
<p>Machine learning tournament prediction model<br>All matches simulated on neutral ground</p>
</div>
<div class="section">
<div class="section-title">Group Stage</div>
<div class="groups">
"""
# Group tables
for group, teams in wc_groups.items():
    sorted_teams = sorted(teams, key=get_points, reverse=True)
    html += f'<div class="group"><h3>Group {group}</h3>'
    for i, team in enumerate(sorted_teams):
        cls = "rank1" if i == 0 else "rank2" if i == 1 else ""
        html += f'<div class="team {cls}"><span>{i+1}. {team}</span><span>{points[team]} pts</span></div>'
    html += '</div>'

html += '</div></div><div class="section"><div class="section-title">Knockout Stage</div><div class="matches">'

# Knockout rounds
def add_round(title, matches, winners):
    global html
    html += f'<div class="round-title">{title}</div>'
    for i, (home, away) in enumerate(matches):
        w = winners[i] if i < len(winners) else "TBD"
        html += f'<div class="match"><span>{home} &nbsp;vs&nbsp; {away}</span><span class="win">{w}</span></div>'

add_round("Round of 32", round_of_32, round_of_32_winners)
add_round("Round of 16", round_of_16, round_of_16_winners)
add_round("Quarter Finals", quarter_finalists, quarter_final_winners)
add_round("Semi Finals", semi_finalists, semi_finalists_winners)
add_round("Final", grand_finalist, grand_finalist_winners)

html += f'</div><div class="final"><div class="trophy">🏆</div><h2>{winner}</h2><p>Predicted 2026 World Cup Champion</p></div></div></body></html>'

with open("wc2026_predictions.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Report saved: wc2026_predictions.html")

html += add_round("Round of 32", round_of_32, round_of_32_winners)
html += add_round("Round of 16", round_of_16, round_of_16_winners)
html += add_round("Quarter Finals", quarter_finalists, quarter_final_winners)
html += add_round("Semi Finals", semi_finalists, semi_finalists_winners)
html += add_round("Final", grand_finalist, grand_finalist_winners)
html += f"<div class='final-winner'>🏆 2026 World Cup Winner: {grand_finalist[0][0] if grand_finalist else 'TBD'}</div>"
html += "</body></html>"

with open("wc2026_predictions.html", "w") as f:
    f.write(html)
print("\nReport saved to wc2026_predictions.html")
    

# Serch Specific Team

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
