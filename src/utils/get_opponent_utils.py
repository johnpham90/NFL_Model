import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def create_defense_teamstat_id(df):
    """
    Add a defense ID column for team stats.
    """
    team_id = []
    for i_game in df.index:
        if df.loc[i_game, 'ts_teamid'] == df.loc[i_game, 'ts_awayteamid']:
            team_id.append(df.loc[i_game, 'ts_hometeamid'])
        else:
            team_id.append(df.loc[i_game, 'ts_awayteamid'])
    df['ts_defenseid'] = team_id
    return df


def create_defense_drivestat_id(df):
    """
    Add a defense ID column for drive stats.
    """
    team_id = []
    for i_game in df.index:
        if df.loc[i_game, 'ds_teamid'] == df.loc[i_game, 'ds_awayteamid']:
            team_id.append(df.loc[i_game, 'ds_hometeamid'])
        else:
            team_id.append(df.loc[i_game, 'ds_awayteamid'])
    df['ds_defenseid'] = team_id
    return df


def fetch_defenseid(season, team_stat, drive_stat):
    """
    Fetch team stats and determine the opponent for each row based on the query results.

    Args:
        season (int): The season year to filter the data.
        stat (str): The statistic to include in the query.

    Returns:
        pd.DataFrame: DataFrame with team stats and opponent team information.
    """
    # Define your query
    query = f"""
       SELECT 
        ts.teamid AS ts_teamid,
        ts.hometeamid AS ts_hometeamid,
        ts.awayteamid AS ts_awayteamid,
        ts.season AS ts_season,
        ts.gamesummaryid AS ts_gamesummaryid,
        ts.{team_stat} AS team_stat,
        ds.teamid AS ds_teamid,
        ds.hometeamid AS ds_hometeamid,
        ds.awayteamid AS ds_awayteamid,
        ds.driveid AS ds_driveid,
        ds.{drive_stat} AS drive_stat
    FROM stats.teamstats ts
    JOIN stats.drivestats ds
    ON ts.gamesummaryid = ds.gamesummaryid
    WHERE ts.season = {season}
    """
    
    # Execute the query 
    query_results = execute_query(query)  

    query_results_teamstat = create_defense_teamstat_id(query_results)
    query_results_drivestat = create_defense_drivestat_id(query_results)
    
    return query_results_teamstat, query_results_drivestat
    
