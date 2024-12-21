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
   
        