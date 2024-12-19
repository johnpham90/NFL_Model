import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query

def situational_features(df):
    '''Create situational-level features'''
    features = {}
    # Add feature engineering logic here
    return features

def red_zone_effiency(data):
    """
    Determine if each drive ended in the red zone.

    Args:
    - data (pd.DataFrame): DataFrame containing 'teamid', 'start_position', and 'net_yards'.

    Returns:
    - pd.DataFrame: Original DataFrame with added 'end_position' and 'red_zone' columns.
    """
    def standardize_position(row):
        """
        Convert field position to a standardized numerical value.
        """
        start_pos = row["start_position"]
        teamid = row["teamid"]

        # Split start position into the team and yard line
        position_team, yard_line = start_pos.split()
        yard_line = int(yard_line)

        # Determine standardized start
        if position_team == teamid:  # Ball is on the team's own side
            return 100 - yard_line
        else:  # Ball is on the opponent's side
            return yard_line

    # Standardize starting position
    data["standardized_start"] = data.apply(standardize_position, axis=1)
    
    # Calculate ending position
    data["end_position"] = data["standardized_start"] - data["net_yards"]
    
    # Determine if the drive ended in the red zone
    data["red_zone"] = data["end_position"].between(1, 20)
    
    return data
