import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

TEAM_DATA = {
    "Australia": {
        "fifa_ranking": 23,
        "qual_wins": 4, "qual_losses": 0, "qual_draws": 0,
        "qual_gf": 10, "qual_ga": 2,
        "form_last10": [3,3,3,3,1,0,3,3,0,3],   # pts per game, most recent first
        "unbeaten_run": 3,
        "wc_appearances": 6,
        "wc_best_result": 2,        # round reached: 1=group, 2=r16, 3=QF, 4=SF, 5=Final, 6=Winner
        "opponent_avg_ranking": 85,
        "clean_sheets_last10": 6,
        "goals_conceded_last10": 4,
        "star_player_rating": 7.2,  # best player avg rating /10
        "squad_depth_rating": 6.8,
        "h2h_wins": 0, "h2h_losses": 2, "h2h_draws": 0,  # vs Turkey specifically
        "home_advantage": 0,        # neutral = 0
        "key_players_available": 1.0,  # 1.0 = full squad
        "avg_goals_scored_qual": 2.5,
        "avg_goals_conceded_qual": 0.5,
        "xg_diff_proxy": 1.8,       # attacking output quality proxy
        "press_intensity": 6.5,     # pressing/tactical rating
        "set_piece_threat": 7.5,    # set piece quality
    },
    "Turkey": {
        "fifa_ranking": 29,
        "qual_wins": 5, "qual_losses": 0, "qual_draws": 2,
        "qual_gf": 14, "qual_ga": 3,
        "form_last10": [3,1,3,3,3,3,1,3,3,3],
        "unbeaten_run": 8,
        "wc_appearances": 3,
        "wc_best_result": 4,        # 3rd place 2002
        "opponent_avg_ranking": 38,
        "clean_sheets_last10": 5,
        "goals_conceded_last10": 6,
        "star_player_rating": 8.5,  # Arda Guler
        "squad_depth_rating": 7.8,
        "h2h_wins": 2, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 0.95,
        "avg_goals_scored_qual": 2.0,
        "avg_goals_conceded_qual": 0.43,
        "xg_diff_proxy": 1.9,
        "press_intensity": 7.2,
        "set_piece_threat": 6.8,
    },
    "Brazil": {
        "fifa_ranking": 5,
        "qual_wins": 12, "qual_losses": 2, "qual_draws": 4,
        "qual_gf": 34, "qual_ga": 14,
        "form_last10": [3,3,1,3,0,3,3,1,3,3],
        "unbeaten_run": 4,
        "wc_appearances": 22,
        "wc_best_result": 6,
        "opponent_avg_ranking": 45,
        "clean_sheets_last10": 4,
        "goals_conceded_last10": 9,
        "star_player_rating": 9.0,  # Vinicius
        "squad_depth_rating": 9.2,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 1.0,
        "avg_goals_scored_qual": 1.9,
        "avg_goals_conceded_qual": 0.78,
        "xg_diff_proxy": 2.3,
        "press_intensity": 7.8,
        "set_piece_threat": 7.5,
    },
    "France": {
        "fifa_ranking": 2,
        "qual_wins": 8, "qual_losses": 0, "qual_draws": 2,
        "qual_gf": 25, "qual_ga": 4,
        "form_last10": [3,3,3,1,3,3,3,1,3,3],
        "unbeaten_run": 6,
        "wc_appearances": 16,
        "wc_best_result": 6,
        "opponent_avg_ranking": 35,
        "clean_sheets_last10": 6,
        "goals_conceded_last10": 5,
        "star_player_rating": 9.3,  # Mbappe
        "squad_depth_rating": 9.5,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 0.95,
        "avg_goals_scored_qual": 2.5,
        "avg_goals_conceded_qual": 0.4,
        "xg_diff_proxy": 2.6,
        "press_intensity": 8.2,
        "set_piece_threat": 8.0,
    },
    "England": {
        "fifa_ranking": 4,
        "qual_wins": 7, "qual_losses": 1, "qual_draws": 2,
        "qual_gf": 22, "qual_ga": 7,
        "form_last10": [3,1,3,3,0,3,3,1,3,3],
        "unbeaten_run": 3,
        "wc_appearances": 16,
        "wc_best_result": 3,
        "opponent_avg_ranking": 38,
        "clean_sheets_last10": 4,
        "goals_conceded_last10": 8,
        "star_player_rating": 8.8,  # Bellingham
        "squad_depth_rating": 9.0,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 0.9,
        "avg_goals_scored_qual": 2.2,
        "avg_goals_conceded_qual": 0.7,
        "xg_diff_proxy": 2.1,
        "press_intensity": 7.5,
        "set_piece_threat": 8.5,
    },
    "Argentina": {
        "fifa_ranking": 1,
        "qual_wins": 11, "qual_losses": 3, "qual_draws": 4,
        "qual_gf": 35, "qual_ga": 18,
        "form_last10": [3,3,0,3,3,1,3,3,0,3],
        "unbeaten_run": 5,
        "wc_appearances": 18,
        "wc_best_result": 6,
        "opponent_avg_ranking": 42,
        "clean_sheets_last10": 4,
        "goals_conceded_last10": 10,
        "star_player_rating": 9.5,  # Messi
        "squad_depth_rating": 9.0,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 1.0,
        "avg_goals_scored_qual": 1.94,
        "avg_goals_conceded_qual": 1.0,
        "xg_diff_proxy": 2.4,
        "press_intensity": 8.0,
        "set_piece_threat": 7.8,
    },
    "Spain": {
        "fifa_ranking": 3,
        "qual_wins": 8, "qual_losses": 0, "qual_draws": 2,
        "qual_gf": 28, "qual_ga": 5,
        "form_last10": [3,3,3,1,3,3,3,3,1,3],
        "unbeaten_run": 9,
        "wc_appearances": 16,
        "wc_best_result": 6,
        "opponent_avg_ranking": 36,
        "clean_sheets_last10": 6,
        "goals_conceded_last10": 5,
        "star_player_rating": 9.0,  # Yamal
        "squad_depth_rating": 9.3,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 1.0,
        "avg_goals_scored_qual": 2.8,
        "avg_goals_conceded_qual": 0.5,
        "xg_diff_proxy": 2.7,
        "press_intensity": 8.5,
        "set_piece_threat": 7.5,
    },
    "Germany": {
        "fifa_ranking": 12,
        "qual_wins": 7, "qual_losses": 1, "qual_draws": 2,
        "qual_gf": 24, "qual_ga": 8,
        "form_last10": [3,1,3,3,0,3,1,3,3,3],
        "unbeaten_run": 2,
        "wc_appearances": 20,
        "wc_best_result": 6,
        "opponent_avg_ranking": 37,
        "clean_sheets_last10": 4,
        "goals_conceded_last10": 8,
        "star_player_rating": 8.5,
        "squad_depth_rating": 8.8,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 0.95,
        "avg_goals_scored_qual": 2.4,
        "avg_goals_conceded_qual": 0.8,
        "xg_diff_proxy": 2.2,
        "press_intensity": 8.0,
        "set_piece_threat": 7.8,
    },
    "USA": {
        "fifa_ranking": 16,
        "qual_wins": 10, "qual_losses": 3, "qual_draws": 5,
        "qual_gf": 28, "qual_ga": 16,
        "form_last10": [3,3,1,0,3,1,3,3,1,3],
        "unbeaten_run": 4,
        "wc_appearances": 11,
        "wc_best_result": 3,
        "opponent_avg_ranking": 55,
        "clean_sheets_last10": 3,
        "goals_conceded_last10": 10,
        "star_player_rating": 8.0,  # Pulisic
        "squad_depth_rating": 7.5,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0.5,      # co-host boost
        "key_players_available": 1.0,
        "avg_goals_scored_qual": 1.56,
        "avg_goals_conceded_qual": 0.89,
        "xg_diff_proxy": 1.6,
        "press_intensity": 7.0,
        "set_piece_threat": 7.0,
    },
    "Portugal": {
        "fifa_ranking": 6,
        "qual_wins": 8, "qual_losses": 0, "qual_draws": 2,
        "qual_gf": 27, "qual_ga": 5,
        "form_last10": [3,3,3,1,3,3,3,3,1,3],
        "unbeaten_run": 7,
        "wc_appearances": 8,
        "wc_best_result": 4,
        "opponent_avg_ranking": 36,
        "clean_sheets_last10": 6,
        "goals_conceded_last10": 5,
        "star_player_rating": 9.2,  # Ronaldo
        "squad_depth_rating": 8.8,
        "h2h_wins": 0, "h2h_losses": 0, "h2h_draws": 0,
        "home_advantage": 0,
        "key_players_available": 1.0,
        "avg_goals_scored_qual": 2.7,
        "avg_goals_conceded_qual": 0.5,
        "xg_diff_proxy": 2.5,
        "press_intensity": 7.8,
        "set_piece_threat": 7.5,
    },
}

def weighted_form(form_last10):
    # Exponentially weight recent games more heavily
    weights = np.exp(-0.2 * np.arange(len(form_last10)))
    weights /= weights.sum()
    return float(np.dot(form_last10, weights))

def build_features(data, opponent=None):
    games = data["qual_wins"] + data["qual_losses"] + data["qual_draws"]
    win_rate = data["qual_wins"] / games
    draw_rate = data["qual_draws"] / games
    gd_per_game = (data["qual_gf"] - data["qual_ga"]) / games
    goals_scored_pg = data["qual_gf"] / games
    goals_conceded_pg = data["qual_ga"] / games

    # Fixture difficulty adjustment
    fix_diff = 100 / data["opponent_avg_ranking"]  # higher = tougher opponents
    adj_win_rate = win_rate * (1 + fix_diff * 0.3)
    adj_gd = gd_per_game * (1 + fix_diff * 0.2)

    # Weighted recent form (recency bias)
    w_form = weighted_form(data["form_last10"])

    # Momentum: unbeaten run + last 3 form
    last3_pts = sum(data["form_last10"][:3])
    momentum = data["unbeaten_run"] * 0.4 + last3_pts * 0.6

    # Defensive rating
    defensive_score = (data["clean_sheets_last10"] / 10) * 10 - (data["goals_conceded_last10"] / 10)

    # H2H record vs this specific opponent
    h2h_total = data["h2h_wins"] + data["h2h_losses"] + data["h2h_draws"]
    h2h_score = (data["h2h_wins"] - data["h2h_losses"]) / max(h2h_total, 1)

    # FIFA rank (inverse — lower number = better)
    rank_score = 100 / data["fifa_ranking"]

    # Tournament pedigree
    pedigree = data["wc_appearances"] * 0.3 + data["wc_best_result"] * 0.7

    return {
        "rank_score":          rank_score,
        "weighted_form":       w_form,
        "momentum":            momentum,
        "adj_win_rate":        adj_win_rate,
        "adj_gd_per_game":     adj_gd,
        "defensive_score":     defensive_score,
        "xg_diff_proxy":       data["xg_diff_proxy"],
        "star_player_rating":  data["star_player_rating"],
        "squad_depth":         data["squad_depth_rating"],
        "fixture_difficulty":  fix_diff,
        "h2h_score":           h2h_score,
        "pedigree":            pedigree,
        "set_piece_threat":    data["set_piece_threat"],
        "press_intensity":     data["press_intensity"],
        "home_advantage":      data["home_advantage"],
        "key_players":         data["key_players_available"],
    }

FEATURE_WEIGHTS = np.array([
    2.5,   # rank_score           — most important single signal
    2.2,   # weighted_form        — who's hot right now
    1.8,   # momentum             — recent run + unbeaten streak
    1.6,   # adj_win_rate         — qual results vs difficulty
    1.5,   # adj_gd_per_game      — winning margin quality
    1.8,   # defensive_score      — defense wins tournaments
    1.4,   # xg_diff_proxy        — attacking quality
    1.3,   # star_player_rating   — matchwinners matter
    1.1,   # squad_depth          — depth for knockout stages
    1.5,   # fixture_difficulty   — who you beat matters
    1.2,   # h2h_score            — direct history
    1.0,   # pedigree             — WC experience
    0.9,   # set_piece_threat     — crucial in tight games
    0.9,   # press_intensity      — tactical quality
    1.0,   # home_advantage       — co-host / nearby support
    0.8,   # key_players          — injury factor
])

def predict_match(team_a: str, team_b: str):
    if team_a not in TEAM_DATA:
        print(f"Unknown team: {team_a}. Available: {list(TEAM_DATA.keys())}")
        return
    if team_b not in TEAM_DATA:
        print(f"Unknown team: {team_b}. Available: {list(TEAM_DATA.keys())}")
        return

    # Build features for all teams (needed for scaling)
    all_feats = {t: build_features(d) for t, d in TEAM_DATA.items()}
    df = pd.DataFrame(all_feats).T
    feature_cols = df.columns.tolist()

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df.values)

    # Synthetic training with domain-weighted logistic boundary
    np.random.seed(42)
    n = 500
    X_train = np.random.randn(n, len(feature_cols))
    log_odds = X_train @ FEATURE_WEIGHTS
    probs = 1 / (1 + np.exp(-log_odds))
    y_train = (np.random.rand(n) < probs).astype(int)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    teams_list = list(TEAM_DATA.keys())
    idx_a = teams_list.index(team_a)
    idx_b = teams_list.index(team_b)

    prob_a = model.predict_proba(X_all[idx_a].reshape(1, -1))[0][1]
    prob_b = model.predict_proba(X_all[idx_b].reshape(1, -1))[0][1]
    total = prob_a + prob_b
    prob_a_norm = round((prob_a / total) * 100, 1)
    prob_b_norm = round((prob_b / total) * 100, 1)

    feats_a = build_features(TEAM_DATA[team_a])
    feats_b = build_features(TEAM_DATA[team_b])

    labels = {
        "rank_score":         "FIFA rank score",
        "weighted_form":      "Weighted form (last 10)",
        "momentum":           "Momentum",
        "adj_win_rate":       "Adj win rate",
        "adj_gd_per_game":    "Adj GD per game",
        "defensive_score":    "Defensive score",
        "xg_diff_proxy":      "xG diff proxy",
        "star_player_rating": "Star player rating",
        "squad_depth":        "Squad depth",
        "fixture_difficulty": "Fixture difficulty",
        "h2h_score":          "H2H score",
        "pedigree":           "WC pedigree",
        "set_piece_threat":   "Set piece threat",
        "press_intensity":    "Press intensity",
        "home_advantage":     "Home advantage",
        "key_players":        "Key players fit",
    }

    print(f"\n{'='*62}")
    print(f"  {team_a.upper()} vs {team_b.upper()}")
    print(f"{'='*62}")
    print(f"\n  {'Metric':<28} {'Wt':>4}  {team_a:<14} {team_b}")
    print(f"  {'-'*58}")
    for i, (k, label) in enumerate(labels.items()):
        a_val = feats_a[k]
        b_val = feats_b[k]
        wt = FEATURE_WEIGHTS[i]
        edge = "<<" if a_val > b_val else ">>" if b_val > a_val else "  "
        print(f"  {label:<28} {wt:>4.1f}  {a_val:<14.3f} {b_val:.3f}  {edge}")

    print(f"\n{'='*62}")
    for team, prob in [(team_a, prob_a_norm), (team_b, prob_b_norm)]:
        bar = "*" * int(prob / 100 * 40)
        print(f"  {team:<20} {prob:5.1f}%  {bar}")
    print(f"{'='*62}")
    winner = team_a if prob_a_norm > prob_b_norm else team_b
    print(f"\n  PREDICTED WINNER: {winner} ({max(prob_a_norm, prob_b_norm)}%)\n")

# ── RUN ───────────────────────────────────────────────────────
print()
predict_match("Australia", "USA")

