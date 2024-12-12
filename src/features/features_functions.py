import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def create_team_features(df):
    '''Create team-level aggregate features'''
    features = {}
    # Add feature engineering logic here
    return features

def create_game_features(df):
    '''Create game-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def prepare_model_data(df):
    '''Prepare final feature set for model'''
    team_features = create_team_features(df)
    game_features = create_game_features(df)
    # Combine features
    return pd.DataFrame()

def generic_team_stats_per_game(season, stat):
    
    query=f"""SELECT *
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results=execute_query(query)

    weeks=np.unique(query_results.loc[:,"week"].values)
    teams=np.unique(query_results.loc[:,"teamid"].values)

    ypg_df=pd.DataFrame(index=weeks, columns=teams)

    for i_week in weeks:
        current_week_idx=np.where(query_results["week"]==i_week)[0]
        current_week_df=pd.DataFrame(query_results.iloc[current_week_idx]).set_index('teamid')
        ypg_df.loc[i_week, current_week_df.index]=current_week_df.loc[:, stat]
        
    average_ypg_df=pd.DataFrame(index=weeks, columns=teams)
    
    for i_week in range(weeks.shape[0]):
        for i_team in teams:
            current_week=weeks[i_week]
            average_ypg_df.loc[current_week,i_team]=np.mean(ypg_df.loc[:current_week, i_team].dropna())
        
    return ypg_df, average_ypg_df
