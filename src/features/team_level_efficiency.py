import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def down_3rd_4th_efficiency(season,stat):
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
    SELECT {stat}, week, teamid
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results = execute_query(query)  # Replace with your DB execution function
    data = pd.DataFrame(query_results, columns=[f'{stat}', 'week', 'teamid'])

    # Parse 'X-Y' format into separate attempts and conversions
    def parse_conversion(conv_str):
        try:
            attempts, conversions = map(int, conv_str.split('-'))
            return attempts, conversions
        except:
            return 0, 0  # Default to zero if parsing fails

    # Apply parsing to create separate columns
    data[['down_attempts', f'{stat}']] = data[f'{stat}'].apply(
        lambda x: pd.Series(parse_conversion(x))
    )


    # Calculate efficiencies safely (avoid division by zero)
    data[f'down_efficiency'] = np.where(
        data['down_attempts'] > 0, 
        data[f'{stat}'] / data['down_attempts'], 
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
                down_pg.loc[week, team] = team_data[f'down_efficiency'].values[0]
                

    # Calculate rolling averages
    for i, week in enumerate(weeks):
        for team in teams:
            down_average.loc[week, team] = down_pg.loc[:week, team].dropna().astype(float).mean()
            
    down_pg=down_pg.shift(1)
    down_average=down_average.shift(1)

    return down_pg, down_average


def passing_efficency(season, stat):
    """This function will caclulate parse the completion data to build completion efficency and passing efficency
    
    completion efficency=completions/attempts
    
    passing efficency=yards/attempts
    Args:
        season (int): what season to calculate stats for
        stat (str): what stat to calculate (completion efficency or passing efficency) 

    Returns:
        pandas dataframe: will return two dataframes, per games and average upto that week
    """    
    
    query = f"""
    SELECT cmp_att_yd_td_int, week, teamid
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results = execute_query(query)  # Replace with your DB execution function
    data = pd.DataFrame(query_results, columns=['cmp_att_yd_td_int', 'week', 'teamid'])
    
    def parse_conversion_passing(conv_str):

        completions, attempts, yards, tds, ints = map(int, conv_str.split('-'))
        return completions, attempts, yards, tds, ints
    data[['completions','attempts', 'yards', 'tds', 'int' ]]=data['cmp_att_yd_td_int'].apply(
        lambda x: pd.Series(parse_conversion_passing(x))
    )



    # Calculate efficiencies safely (avoid division by zero)
    
    if stat=="completion efficiency":
        data[f'{stat}'] = np.where(
            data[f'attempts'] > 0, 
            data['completions'] / data['attempts'], 
            np.nan
        )
    if stat=="passing efficiency":
        data[f'{stat}'] = np.where(
            data[f'attempts'] > 0, 
            data['yards'] / data['attempts'], 
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
                down_pg.loc[week, team] = team_data[f'{stat}'].values[0]
                

    # Calculate rolling averages
    for i, week in enumerate(weeks):
        for team in teams:
            down_average.loc[week, team] = down_pg.loc[:week, team].dropna().astype(float).mean()
        

    down_pg=down_pg.shift(1)
    down_average=down_average.shift(1)
    return down_pg, down_average
   
        