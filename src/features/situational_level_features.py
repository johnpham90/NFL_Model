import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def situational_features(df):
    '''Create situational-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def down_3rd_4th_efficiency():
    """
    Calculate 3rd and 4th down efficiency rates for each team per week 
    and their rolling averages for the season.
    
    Returns:
        dict: Contains DataFrames for:
            - 'third_down_pg': Weekly 3rd down efficiency per game.
            - 'third_down_average': Rolling average 3rd down efficiency.
            - 'fourth_down_pg': Weekly 4th down efficiency per game.
            - 'fourth_down_average': Rolling average 4th down efficiency.
    """
    # Query to fetch team stats
    query = """
    SELECT third_down_conv, fourth_down_conv, week, teamid
    FROM stats.teamstats
    WHERE season = '2024'
    """
    query_results = execute_query(query)  # Replace with your DB execution function
    data = pd.DataFrame(query_results, columns=['third_down_conv', 'fourth_down_conv', 'week', 'teamid'])

    # Parse 'X-Y' format into separate attempts and conversions
    def parse_conversion(conv_str):
        try:
            attempts, conversions = map(int, conv_str.split('-'))
            return attempts, conversions
        except:
            return 0, 0  # Default to zero if parsing fails

    # Apply parsing to create separate columns
    data[['third_down_attempts', 'third_down_conversions']] = data['third_down_conv'].apply(
        lambda x: pd.Series(parse_conversion(x))
    )
    data[['fourth_down_attempts', 'fourth_down_conversions']] = data['fourth_down_conv'].apply(
        lambda x: pd.Series(parse_conversion(x))
    )

    # Calculate efficiencies safely (avoid division by zero)
    data['third_down_efficiency'] = np.where(
        data['third_down_attempts'] > 0, 
        data['third_down_conversions'] / data['third_down_attempts'], 
        np.nan
    )
    data['fourth_down_efficiency'] = np.where(
        data['fourth_down_attempts'] > 0, 
        data['fourth_down_conversions'] / data['fourth_down_attempts'], 
        np.nan
    )

    # Get unique weeks and teams
    weeks = np.unique(data['week'].values)
    teams = np.unique(data['teamid'].values)

    # Initialize DataFrames for weekly and rolling averages
    third_down_pg = pd.DataFrame(index=weeks, columns=teams)
    fourth_down_pg = pd.DataFrame(index=weeks, columns=teams)

    third_down_average = pd.DataFrame(index=weeks, columns=teams)  # Rolling 3rd down average per game
    fourth_down_average = pd.DataFrame(index=weeks, columns=teams)  # Rolling 4th down average per game

    # Populate weekly stats
    for week in weeks:
        current_week_data = data[data['week'] == week]
        for team in teams:
            team_data = current_week_data[current_week_data['teamid'] == team]
            if not team_data.empty:
                third_down_pg.loc[week, team] = team_data['third_down_efficiency'].values[0]
                fourth_down_pg.loc[week, team] = team_data['fourth_down_efficiency'].values[0]

    # Calculate rolling averages
    for i, week in enumerate(weeks):
        for team in teams:
            third_down_average.loc[week, team] = third_down_pg.loc[:week, team].astype(float).mean()
            fourth_down_average.loc[week, team] = fourth_down_pg.loc[:week, team].astype(float).mean()

    # Return results as a dictionary
    return {
        'third_down_pg': third_down_pg,
        'third_down_average': third_down_average,
        'fourth_down_pg': fourth_down_pg,
        'fourth_down_average': fourth_down_average
    }
   
        