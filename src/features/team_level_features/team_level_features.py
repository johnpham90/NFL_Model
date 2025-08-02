import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def team_features(df):
    '''Create historical-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def offensive_mettrics(season, stat):
    
    query=f"""SELECT *
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results=execute_query(query)

    weeks=np.unique(query_results.loc[:,"week"].values)
    teams=np.unique(query_results.loc[:,"teamid"].values)

    stats_pg_df=pd.DataFrame(index=weeks, columns=teams)

    for i_week in weeks:
        current_week_idx=np.where(query_results["week"]==i_week)[0]
        current_week_df=pd.DataFrame(query_results.iloc[current_week_idx]).set_index('teamid')
        stats_pg_df.loc[i_week, current_week_df.index]=current_week_df.loc[:, stat]
        
    average_stats_pg_df=pd.DataFrame(index=weeks, columns=teams)
    
    for i_week in range(weeks.shape[0]):
        for i_team in teams:
            current_week=weeks[i_week]
            average_stats_pg_df.loc[current_week,i_team]=np.mean(stats_pg_df.loc[:current_week, i_team].dropna())
        
    return stats_pg_df, average_stats_pg_df


def defensive_metrics(season, stat):
    """
    Calculate yards allowed per game (YPG) and the rolling average yards allowed throughout the season
    based on the specified stat, from the defensive team's perspective.

    Args:
    - season (int): The season to query.
    - stat (str): The stat to calculate (e.g., 'total_yards', 'passing_yards').

    Returns:
    - pd.DataFrame: Yards allowed for each game by week and team.
    - pd.DataFrame: Rolling average yards allowed per game for each team.
    """
    query = f"""
    SELECT *
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results = execute_query(query)

    # Ensure data is sorted by team and week for rolling averages
    query_results = query_results.sort_values(by=["teamid", "week"])

    # Add a column for the defensive team's metrics
    def assign_defensive_stats(row):
        if row["teamid"] == row["hometeamid"]:
            # If the stat belongs to the home team, the defense is the away team
            defensive_team = row["awayteamid"]
        elif row["teamid"] == row["awayteamid"]:
            # If the stat belongs to the away team, the defense is the home team
            defensive_team = row["hometeamid"]
        else:
            raise ValueError("Invalid teamid comparison")
        
        # Return the stat as the defensive metric for the opposing team
        return defensive_team, row[stat]

    # Apply the logic to calculate defensive metrics
    query_results[["defensive_team", "stats_allowed"]] = query_results.apply(
        lambda row: assign_defensive_stats(row), axis=1, result_type="expand"
    )

    # Aggregate stats by defensive team
    defensive_stats_df = query_results.pivot(index="week", columns="defensive_team", values="stats_allowed")

    # Calculate rolling average of stats allowed
    query_results["stats_allowed_avg"] = (
        query_results.groupby("defensive_team")["stats_allowed"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    average_pg_df = query_results.pivot(index="week", columns="defensive_team", values="stats_allowed_avg")

    return defensive_stats_df, average_pg_df
