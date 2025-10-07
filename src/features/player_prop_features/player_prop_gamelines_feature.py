"""Game lines feature engineering for player props.

Adds betting context features:
- Spread and over/under totals
- Implied team totals
- Rest and bye week indicators
"""
import numpy as np
import pandas as pd
from src.utils.db_utils import execute_query


def add_game_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join game lines and calculate derived betting features.
    
    Args:
        df: Player game data with gamesummaryid
        
    Returns:
        DataFrame with game line features added:
        - team_implied_total: Expected points for player's team
        - opp_implied_total: Expected points for opponent
        - game_total: Total expected points (O/U)
        - is_favorite: Binary indicator if team is favored
        - spread_magnitude: Absolute value of spread
        - days_rest: Days since last game
        - weeks_from_bye: Weeks since bye week
        
    Example:
        >>> df = load_player_data(2020, "pass_yds")
        >>> df = add_game_lines(df)
        >>> print(df['team_implied_total'].mean())
    """
    lines_query = """
    SELECT 
        gamesummaryid,
        spread,
        over_under as total,
        hometeamid,
        awayteamid,
        spreadfavoriteteam,
        days_since_last_game_hometeam,
        days_since_last_game_awayteam,
        hometeam_weeks_from_bye,
        awayteam_weeks_from_bye
    FROM stats.gamesummary
    WHERE spread IS NOT NULL AND over_under IS NOT NULL
    """
    lines = execute_query(lines_query)
    
    if lines is None or lines.empty:
        print("[WARNING] No game lines data available - skipping game context features")
        return df
    
    # Calculate implied team totals
    # Formula: home_total = (total - spread) / 2, away_total = (total + spread) / 2
    # Example: O/U 48, spread -7 (home favored) -> home 27.5, away 20.5
    lines['home_implied_total'] = (lines['total'] - lines['spread']) / 2
    lines['away_implied_total'] = (lines['total'] + lines['spread']) / 2
    
    # Merge to player data
    original_count = len(df)
    df = df.merge(lines, on='gamesummaryid', how='left', suffixes=('', '_lines'))
    
    if len(df) != original_count:
        print(f"[WARNING] Merge changed row count: {original_count} -> {len(df)}")
    
    # Calculate player's team implied total
    df['team_implied_total'] = np.where(
        df['teamid'] == df['hometeamid_lines'],
        df['home_implied_total'],
        df['away_implied_total']
    )
    
    df['opp_implied_total'] = np.where(
        df['teamid'] == df['hometeamid_lines'],
        df['away_implied_total'],
        df['home_implied_total']
    )
    
    # Is team favored?
    df['is_favorite'] = (df['teamid'] == df['spreadfavoriteteam']).astype(int)
    
    # Spread magnitude (absolute value)
    df['spread_magnitude'] = np.abs(df['spread'])
    
    # Game total (O/U)
    df['game_total'] = df['total']
    
    # Rest and bye week features
    df['days_rest'] = np.where(
        df['teamid'] == df['hometeamid_lines'],
        df['days_since_last_game_hometeam'],
        df['days_since_last_game_awayteam']
    )
    
    df['weeks_from_bye'] = np.where(
        df['teamid'] == df['hometeamid_lines'],
        df['hometeam_weeks_from_bye'],
        df['awayteam_weeks_from_bye']
    )
    
    # Clean up temporary columns
    cols_to_drop = [
        'hometeamid_lines', 'awayteamid_lines', 'spread', 'total',
        'home_implied_total', 'away_implied_total', 'spreadfavoriteteam',
        'days_since_last_game_hometeam', 'days_since_last_game_awayteam',
        'hometeam_weeks_from_bye', 'awayteam_weeks_from_bye'
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    
    # Report on missing data
    missing_lines = df['team_implied_total'].isna().sum()
    if missing_lines > 0:
        print(f"[WARNING] {missing_lines} records missing game line data")
    else:
        print(f"[INFO] Added game line features to {len(df)} records")
    
    return df


__all__ = ['add_game_lines']