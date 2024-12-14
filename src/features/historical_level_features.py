import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query


def fetch_head_to_head_data():
    """
    Query head-to-head data from the database and process it into a DataFrame.
    """
    query = """
    SELECT awayteamid AS away_team, hometeamid AS home_team, awayscore, homescore, date
    FROM stats.gamesummary
    WHERE season = '2024'
    """
    # Execute the query and fetch the data
    df = execute_query(query)  # Replace with your database query function
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_h2h_features(df, away_team, home_team, num_games=5):
    """
    Calculate head-to-head features between the away and home teams.

    Args:
        df (pd.DataFrame): DataFrame containing historical game data.
        away_team (str): Away team ID.
        home_team (str): Home team ID.
        num_games (int): Number of recent games to consider.

    Returns:
        dict: Head-to-head features.
    """
    # Filter historical matchups between away_team and home_team
    h2h_games = df[
        ((df["away_team"] == away_team) & (df["home_team"] == home_team)) |
        ((df["away_team"] == home_team) & (df["home_team"] == away_team))
    ].sort_values("date", ascending=False)
    
    # Limit to recent games
    h2h_games = h2h_games.head(num_games)
    
    if h2h_games.empty:
        return {
            "h2h_games_played": 0,
            "h2h_away_wins": 0,
            "h2h_home_wins": 0,
            "h2h_away_avg_points": 0,
            "h2h_home_avg_points": 0,
            "h2h_away_streak": 0
        }
    
    # Calculate metrics
    away_wins = sum(
        ((h2h_games["away_team"] == away_team) & (h2h_games["awayscore"] > h2h_games["homescore"])) |
        ((h2h_games["home_team"] == away_team) & (h2h_games["homescore"] > h2h_games["awayscore"]))
    )
    home_wins = len(h2h_games) - away_wins
    away_avg_points = h2h_games.apply(
        lambda row: row["awayscore"] if row["away_team"] == away_team else row["homescore"], axis=1
    ).mean()
    home_avg_points = h2h_games.apply(
        lambda row: row["homescore"] if row["home_team"] == home_team else row["awayscore"], axis=1
    ).mean()
    
    # Calculate streak for away_team
    away_streak = 0
    for _, row in h2h_games.iterrows():
        if (row["away_team"] == away_team and row["awayscore"] > row["homescore"]) or \
           (row["home_team"] == away_team and row["homescore"] > row["awayscore"]):
            away_streak += 1
        else:
            break
    
    # Calculate home streak
    home_streak = 0
    for _, row in h2h_games.iterrows():
        if (row["home_team"] == home_team and row["homescore"] > row["awayscore"]) or \
           (row["away_team"] == home_team and row["awayscore"] > row["homescore"]):
            home_streak += 1
        else:
            break
    
    return {
        "h2h_games_played": len(h2h_games),
        "h2h_away_wins": away_wins,
        "h2h_home_wins": home_wins,
        "h2h_away_avg_points": away_avg_points,
        "h2h_home_avg_points": home_avg_points,
        "h2h_away_streak": away_streak,
        "h2h_home_streak": home_streak
    }