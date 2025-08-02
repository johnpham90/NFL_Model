import os
import sys
notebook_dir = os.path.abspath(os.path.dirname(''))
project_root = os.path.dirname(notebook_dir)
sys.path.append(project_root)
import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query


def build_spread_variable(season):
    """This function will bill the spread variable which will be used as a y variable in the machine learning code. 
    spread=home_score-away_score

    Args:
        season (int): season you want to build a spread for

    Returns:
        dict: returns a dictionary where the keys are the week, first item is the home teams, second item is the away teams, and third item is the spread
    """  
    query = f"""
    SELECT *
    FROM stats.gamesummary
    WHERE season = {season}
    """
    tables_df = execute_query(query)
    
    weeks=np.unique(tables_df["week"])
    y_out_dict={}
    for i_week in weeks:
        current_week_games_idx=np.where(tables_df["week"]==i_week)[0]
        current_week_games=tables_df.iloc[current_week_games_idx]
        
        home_team_list=[]
        away_team_list=[]
        actual_spread_list=[]
        
        for i_game in current_week_games.index:
            home_team_list.append(current_week_games.loc[i_game, 'hometeamid'])
            away_team_list.append(current_week_games.loc[i_game, 'awayteamid'])
            actual_spread=current_week_games.loc[i_game, 'homescore']-current_week_games.loc[i_game, 'awayscore']
            actual_spread_list.append(actual_spread)
            
        y_out_dict[i_week]=[home_team_list, away_team_list, actual_spread_list]
        
    return y_out_dict