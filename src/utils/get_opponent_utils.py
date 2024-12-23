import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def create_defense_team_id(df):
    team_id=[]

    for i_game in df.index:
        if df.loc[i_game, 'teamid']==df.loc[i_game, 'awayteamid']:
            team_id.append(df.loc[i_game, 'hometeamid'])
        else:
            team_id.append(df.loc[i_game, 'awayteamid'])
    df['defenseid']=team_id

    return df

def fetch_and_get_opponents(season, stat, table):
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
    SELECT *
    FROM stats.teamstats ts
    JOIN stats.drivestats ds
    ON ts.gamesummaryid = ds.gamesummaryid
    WHERE season = {season}
    """
    
    # Execute the query (replace with your actual database function)
    query_results = execute_query(query)  # Replace with actual DB execution function
    query_results_process = create_defense_team_id(query_results)

    return query_results_process
    
