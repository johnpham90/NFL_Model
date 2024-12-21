import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def situational_features(df):
    '''Create situational-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def down_3rd_4th_efficiency(season,down):
    """
    Calculate 3rd and 4th down efficiency rates for each team per week 
    and their rolling averages for the season.
    
    Returns:
        dict: Contains DataFrames for:
            - 'down_pg': Weekly 3rd/4th down efficiency per game.
            - 'down_average': Rolling average 3rd/4th down efficiency.

    """
    # Query to fetch team stats
    query = f"""
    SELECT {down}_down_conv, week, teamid
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results = execute_query(query)  # Replace with your DB execution function
    data = pd.DataFrame(query_results, columns=[f'{down}_down_conv', 'week', 'teamid'])

    # Parse 'X-Y' format into separate attempts and conversions
    def parse_conversion(conv_str):
        try:
            attempts, conversions = map(int, conv_str.split('-'))
            return attempts, conversions
        except:
            return 0, 0  # Default to zero if parsing fails

    # Apply parsing to create separate columns
    data[[f'{down}_down_attempts', f'{down}_down_conversions']] = data[f'{down}_down_conv'].apply(
        lambda x: pd.Series(parse_conversion(x))
    )


    # Calculate efficiencies safely (avoid division by zero)
    data[f'{down}_down_efficiency'] = np.where(
        data[f'{down}_down_attempts'] > 0, 
        data[f'{down}_down_conversions'] / data[f'{down}_down_attempts'], 
        np.nan
    )


    # Get unique weeks and teams
    weeks = np.unique(data['week'].values)
    teams = np.unique(data['teamid'].values)

    # Initialize DataFrames for weekly and rolling averages
    down_pg = pd.DataFrame(index=weeks, columns=teams)


    down_average = pd.DataFrame(index=weeks, columns=teams)  # Rolling 3rd down average per game
     

    # Populate weekly stats
    for week in weeks:
        current_week_data = data[data['week'] == week]
        for team in teams:
            team_data = current_week_data[current_week_data['teamid'] == team]
            if not team_data.empty:
                down_pg.loc[week, team] = team_data[f'{down}_down_efficiency'].values[0]
                

    # Calculate rolling averages
    for i, week in enumerate(weeks):
        for team in teams:
            down_average.loc[week, team] = down_pg.loc[:week, team].astype(float).mean()
            


    return down_pg, down_average
   
        