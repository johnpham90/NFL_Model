import pandas as pd
import numpy as np
def add_game_week_qid(df, season_col='season', week_col='week', qid_col='qid'):
    """
    Add a query ID (qid) for each unique combination of season and week.
    All games in the same season and week will have the same qid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing NFL game data
    season_col : str, default 'season'
        Name of the column containing season information
    week_col : str, default 'week'
        Name of the column containing week information
    qid_col : str, default 'qid'
        Name of the new column to create for query IDs
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added qid column
        
    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'season': [2023, 2023, 2023, 2024, 2024, 2024],
    ...     'week': [1, 1, 2, 1, 1, 2],
    ...     'team': ['A', 'B', 'C', 'D', 'E', 'F']
    ... })
    >>> df_with_qid = add_game_week_qid(df)
    >>> print(df_with_qid)
       season  week team  qid
    0    2023     1    A    0
    1    2023     1    B    0
    2    2023     2    C    1
    3    2024     1    D    2
    4    2024     1    E    2
    5    2024     2    F    3
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Check if required columns exist
    if season_col not in df_copy.columns:
        raise ValueError(f"Column '{season_col}' not found in DataFrame")
    if week_col not in df_copy.columns:
        raise ValueError(f"Column '{week_col}' not found in DataFrame")
    
    # Create a unique identifier for each season-week combination
    # Sort by season and week to ensure chronological order
    unique_combinations = (df_copy[[season_col, week_col]]
                          .drop_duplicates()
                          .sort_values([season_col, week_col])
                          .reset_index(drop=True))
    
    # Assign sequential qid values
    unique_combinations[qid_col] = range(len(unique_combinations))
    
    # Merge back to original DataFrame
    df_with_qid = df_copy.merge(
        unique_combinations, 
        on=[season_col, week_col], 
        how='left'
    )
    
    # Ensure qid is integer type
    df_with_qid[qid_col] = df_with_qid[qid_col].astype(int)
    
    print(f"Added {qid_col} column with {df_with_qid[qid_col].nunique()} unique season-week combinations")
    print(f"QID range: {df_with_qid[qid_col].min()} to {df_with_qid[qid_col].max()}")
    
    return df_with_qid