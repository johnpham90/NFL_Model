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


def defensive_metrics(query_results, stat):
    """
    Calculate defensive stats based on the opponent's offensive stats.

    Args:
        query_results (pd.DataFrame): DataFrame with team stats and defense IDs.
        stat (str): The statistic to calculate (e.g., "yards").

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Per-game defensive stats and rolling averages.
    """
    # Ensure the query_results DataFrame includes necessary columns
    if 'defenseid' not in query_results.columns:
        raise ValueError("The input DataFrame must include a 'defenseid' column.")

    # Extract unique weeks and teams
    weeks = query_results["week"].unique()
    teams = query_results["teamid"].unique()

    # Initialize DataFrame for defensive stats
    defensive_stats_pg_df = pd.DataFrame(index=weeks, columns=teams)

    # Calculate per-game defensive stats
    for week in weeks:
        current_week_data = query_results[query_results["week"] == week]
        for team in teams:
            # Find rows where the `defenseid` matches the team
            defense_rows = current_week_data[current_week_data["defenseid"] == team]
            if not defense_rows.empty:
                # Sum the `stat` column for the opposing team's offensive stats
                defensive_stats_pg_df.loc[week, team] = defense_rows[stat].sum()

    # Calculate rolling averages for defensive stats
    avg_defensive_stats_pg_df = pd.DataFrame(index=weeks, columns=teams)
    for week in weeks:
        for team in teams:
            avg_defensive_stats_pg_df.loc[week, team] = defensive_stats_pg_df.loc[:week, team].dropna().mean()

    return defensive_stats_pg_df, avg_defensive_stats_pg_df

