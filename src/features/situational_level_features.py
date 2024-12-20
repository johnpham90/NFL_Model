import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def situational_features(df):
    '''Create situational-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def red_zone_efficiency(season):
    """
    Calculate red zone stats (attempts, scores, efficiency) per week, 
    and provide them in a pivoted format (weeks as rows, teams as columns).
    
    Args:
    - season (int): The season to calculate red zone metrics for.

    Returns:
    - dict: Dictionary containing all metrics:
        - attempts_pg: Redzone attempts per game
        - scores_pg: Redzone touchdowns per game
        - efficiency_pg: Touchdown percentage per game
        - rolling_attempts_avg: Running average of attempts
        - rolling_scores_avg: Running average of touchdowns
        - rolling_efficiency_avg: Running average of efficiency
    """
    # Query data
    query = f"""
    SELECT teamid, gamesummaryid, week, start_at, net_yds, end_event
    FROM stats.drivestats
    WHERE season = {season}
    """
    query_results = execute_query(query)

    # Initialize list to hold per-game stats
    game_stats = []

    # Group data by gamesummaryid and teamid
    grouped = query_results.groupby(["gamesummaryid", "teamid"])

    for (gamesummaryid, teamid), group in grouped:
        # Filter out rows with missing or invalid start_at
        valid_group = group[group["start_at"].notna()]
        valid_group = valid_group[valid_group["start_at"].str.contains(" ")]

        red_zone_attempts = 0
        red_zone_scores = 0

        for idx, row in valid_group.iterrows():
            try:
                # Parse team and yard line from start_at
                position_team, yard_line = row["start_at"].split()
                yard_line = int(yard_line)

                # Determine standardized starting position
                if position_team == row["teamid"]:
                    start_yards = 100 - yard_line  # Own side
                else:
                    start_yards = yard_line  # Opponent's side

                # Calculate ending position
                end_position = start_yards - row["net_yds"]

                # Check if drive ends in or passes through the red zone (including TDs)
                if end_position <= 20:  # Modified to include touchdowns (end_position <= 0)
                    red_zone_attempts += 1
                    if row["end_event"] == "Touchdown":
                        red_zone_scores += 1

            except Exception as e:
                print(f"Error processing row: {row}, Error: {e}")

        # Calculate efficiency (avoid division by zero)
        red_zone_efficiency = red_zone_scores / red_zone_attempts if red_zone_attempts > 0 else 0

        game_stats.append({
            "gamesummaryid": gamesummaryid,
            "teamid": teamid,
            "week": group["week"].iloc[0],
            "red_zone_attempts": red_zone_attempts,
            "red_zone_scores": red_zone_scores,
            "red_zone_efficiency": red_zone_efficiency
        })

    # Convert per-game stats into a DataFrame
    game_stats_df = pd.DataFrame(game_stats)

    # Pivot the data for per-week stats
    attempts_pg_df = game_stats_df.pivot(index="week", columns="teamid", values="red_zone_attempts")
    scores_pg_df = game_stats_df.pivot(index="week", columns="teamid", values="red_zone_scores")
    efficiency_pg_df = game_stats_df.pivot(index="week", columns="teamid", values="red_zone_efficiency")

    # Calculate rolling averages and pivot
    game_stats_df["rolling_attempts"] = game_stats_df.groupby("teamid")["red_zone_attempts"].expanding().mean().reset_index(level=0, drop=True)
    game_stats_df["rolling_scores"] = game_stats_df.groupby("teamid")["red_zone_scores"].expanding().mean().reset_index(level=0, drop=True)
    game_stats_df["rolling_efficiency"] = game_stats_df.groupby("teamid")["red_zone_efficiency"].expanding().mean().reset_index(level=0, drop=True)

    rolling_attempts_avg_df = game_stats_df.pivot(index="week", columns="teamid", values="rolling_attempts")
    rolling_scores_avg_df = game_stats_df.pivot(index="week", columns="teamid", values="rolling_scores")
    rolling_efficiency_avg_df = game_stats_df.pivot(index="week", columns="teamid", values="rolling_efficiency")

    # Return all metrics in a single dictionary
    return {
        "attempts_pg": attempts_pg_df,
        "scores_pg": scores_pg_df,
        "efficiency_pg": efficiency_pg_df,
        "rolling_attempts_avg": rolling_attempts_avg_df,
        "rolling_scores_avg": rolling_scores_avg_df,
        "rolling_efficiency_avg": rolling_efficiency_avg_df
    }