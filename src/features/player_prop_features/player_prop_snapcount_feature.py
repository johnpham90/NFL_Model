"""
Snap count and usage rate features for player props.

Provides player workload and role indicators from stats.snapcounts:
- Offensive snap percentage (off_pct)
- Snap count trends (increasing/decreasing role)
- Starter identification (high snap % players)
- Role consistency (snap volatility)
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from src.utils.db_utils import get_connection
from sqlalchemy import text


def load_snap_counts(start_season: int = 2018) -> pd.DataFrame:
    """
    Load snap count data from stats.snapcounts table.
    
    Args:
        start_season: First season to include
        
    Returns:
        DataFrame with columns:
        - playerid, gamesummaryid, teamid, season, week, pos
        - off_num, off_snap_pct (offensive snaps)
        - def_num, def_snap_pct (defensive snaps) 
        - st_num, st_snap_pct (special teams snaps)
    """
    query = f"""
        SELECT 
            playerid,
            gamesummaryid,
            teamid,
            season,
            week,
            pos,
            player,
            
            -- Offensive snaps
            off_num,
            CAST(REPLACE(off_pct, '%', '') AS FLOAT) / 100.0 as off_snap_pct,
            
            -- Defensive snaps  
            def_num,
            CAST(REPLACE(def_pct, '%', '') AS FLOAT) / 100.0 as def_snap_pct,
            
            -- Special teams snaps
            st_num,
            CAST(REPLACE(st_pct, '%', '') AS FLOAT) / 100.0 as st_snap_pct
            
        FROM stats.snapcounts
        WHERE season >= {start_season}
          AND off_pct IS NOT NULL
    """
    
    engine = get_connection()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    print(f"[INFO] Loaded {len(df)} snap count records from season {start_season}")
    return df


def add_snap_count_features(
    player_df: pd.DataFrame,
    position: str,
    min_periods: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add snap count features to player prop dataframe.
    
    Features created:
    - off_snap_pct_hist: Historical average offensive snap %
    - off_snap_pct_roll3: Rolling 3-game average snap %
    - snap_trend: Change in snap % (recent vs historical)
    - snap_volatility: Standard deviation of snap % (role consistency)
    - is_starter: Binary indicator if player is regular starter (>60% snaps)
    
    Args:
        player_df: Player-game dataframe with playerid, gamesummaryid, season, week
        position: Position type (QB, RB, PASS_CATCHER)
        min_periods: Minimum games required for rolling calculations
        
    Returns:
        Tuple of (updated dataframe, list of new feature names)
    """
    print(f"Adding snap count features for {position}...")
    
    # Load snap count data
    start_season = player_df['season'].min()
    snap_df = load_snap_counts(start_season)
    
    # Filter to relevant offensive positions
    if position == 'QB':
        snap_df = snap_df[snap_df['pos'] == 'QB']
    elif position == 'RB':
        snap_df = snap_df[snap_df['pos'] == 'RB']
    elif position == 'PASS_CATCHER':
        snap_df = snap_df[snap_df['pos'].isin(['WR', 'TE'])]
    else:
        print(f"[WARNING] Unknown position: {position}")
        return player_df, []
    
    # Sort for proper temporal ordering
    snap_df = snap_df.sort_values(['playerid', 'season', 'week'])
    
    # Calculate historical average offensive snap % (EXCLUDE current game to prevent leakage)
    snap_df['off_snap_pct_hist'] = (
        snap_df.groupby('playerid')['off_snap_pct']
        .shift(1)  # Exclude current game
        .expanding(min_periods=min_periods)
        .mean()
        .reset_index(drop=True)
    )
    
    # Calculate rolling 3-game average
    snap_df['off_snap_pct_roll3'] = (
        snap_df.groupby('playerid')['off_snap_pct']
        .shift(1)
        .rolling(window=3, min_periods=min_periods)
        .mean()
        .reset_index(drop=True)
    )
    
    # Calculate snap trend (recent 3 games vs earlier 3 games)
    snap_df['off_snap_pct_prev3'] = (
        snap_df.groupby('playerid')['off_snap_pct']
        .shift(4)  # Games 4-6 ago
        .rolling(window=3, min_periods=2)
        .mean()
        .reset_index(drop=True)
    )
    
    snap_df['snap_trend'] = (
        snap_df['off_snap_pct_roll3'] - snap_df['off_snap_pct_prev3']
    )
    
    # Calculate snap volatility (role consistency)
    snap_df['snap_volatility'] = (
        snap_df.groupby('playerid')['off_snap_pct']
        .shift(1)
        .rolling(window=5, min_periods=3)
        .std()
        .reset_index(drop=True)
    )
    
    # Identify regular starters (>60% offensive snaps historically)
    snap_df['is_starter'] = (snap_df['off_snap_pct_hist'] > 0.60).astype(int)
    
    # Merge snap features to player data
    snap_features = [
        'off_snap_pct_hist',
        'off_snap_pct_roll3',
        'snap_trend',
        'snap_volatility',
        'is_starter'
    ]
    
    merge_cols = ['playerid', 'gamesummaryid', 'season', 'week'] + snap_features
    
    player_df = player_df.merge(
        snap_df[merge_cols],
        on=['playerid', 'gamesummaryid', 'season', 'week'],
        how='left'
    )
    
    # Fill missing values (players not in snap count data)
    # Likely means they didn't play (injury/inactive)
    for col in snap_features:
        if col in player_df.columns:
            if col == 'is_starter':
                player_df[col] = player_df[col].fillna(0)  # Not a starter if no data
            else:
                player_df[col] = player_df[col].fillna(0)  # No snaps if no data
    
    print(f"[SUCCESS] Added {len(snap_features)} snap count features")
    
    return player_df, snap_features


def add_usage_rate_features(
    player_df: pd.DataFrame,
    position: str,
    min_periods: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add usage rate features (targets, carries, touches per game).
    
    Only applicable for positions with volume stats (QB, RB, WR, TE).
    
    Args:
        player_df: Player-game dataframe
        position: Position type
        min_periods: Minimum games for rolling calculations
        
    Returns:
        Tuple of (updated dataframe, list of new feature names)
    """
    print(f"Adding usage rate features for {position}...")
    
    features_added = []
    
    # QB usage = pass attempts
    if position == 'QB' and 'pass_att' in player_df.columns:
        player_df = player_df.sort_values(['playerid', 'season', 'week'])
        
        player_df['pass_att_hist'] = (
            player_df.groupby('playerid')['pass_att']
            .shift(1)
            .expanding(min_periods=min_periods)
            .mean()
            .reset_index(drop=True)
        )
        features_added.append('pass_att_hist')
    
    # RB usage = carries + targets
    elif position == 'RB':
        if 'rush_att' in player_df.columns and 'targets' in player_df.columns:
            player_df['total_touches'] = player_df['rush_att'] + player_df['targets']
            
            player_df = player_df.sort_values(['playerid', 'season', 'week'])
            
            player_df['total_touches_hist'] = (
                player_df.groupby('playerid')['total_touches']
                .shift(1)
                .expanding(min_periods=min_periods)
                .mean()
                .reset_index(drop=True)
            )
            features_added.append('total_touches_hist')
    
    # PASS_CATCHER usage = targets (applies to RB/WR/TE for receiving props)
    elif position == 'PASS_CATCHER' and 'targets' in player_df.columns:
        player_df = player_df.sort_values(['playerid', 'season', 'week'])
        
        player_df['targets_hist'] = (
            player_df.groupby('playerid')['targets']
            .shift(1)
            .expanding(min_periods=min_periods)
            .mean()
            .reset_index(drop=True)
        )
        
        # Also calculate target share (% of team targets)
        team_targets = player_df.groupby(['teamid', 'season', 'week'])['targets'].transform('sum')
        player_df['target_share'] = np.where(
            team_targets > 0,
            player_df['targets'] / team_targets,
            0
        )
        
        player_df['target_share_hist'] = (
            player_df.groupby('playerid')['target_share']
            .shift(1)
            .expanding(min_periods=min_periods)
            .mean()
            .reset_index(drop=True)
        )
        
        features_added.extend(['targets_hist', 'target_share_hist'])
    
    if features_added:
        print(f"[SUCCESS] Added {len(features_added)} usage rate features")
    else:
        print(f"[INFO] No usage features added for {position}")
    
    return player_df, features_added


__all__ = [
    'load_snap_counts',
    'add_snap_count_features',
    'add_usage_rate_features'
]
