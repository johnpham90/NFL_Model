"""
Opponent defense and matchup history features for player props.

Provides:
- Opponent defensive performance (yards allowed by position)
- Opponent defensive pressure metrics (aggregated from stats.defense_advanced)
- Player vs opponent historical performance
- Historical matchup counts
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from src.config.player_prop_config import PropType, PROP_TYPE_TO_TARGET_COL, PROP_TYPE_TO_DEFENSE_TARGET
from src.utils.db_utils import execute_query


def add_opponent_defense_features(
    df: pd.DataFrame,
    prop_type: PropType
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add opponent defensive stats (yards allowed by position).
    
    Aggregates how many yards the opponent defense has allowed to this
    position group historically. Uses proper temporal ordering to prevent
    data leakage.
    
    Args:
        df: Player game dataframe with opp_teamid, season, week
        prop_type: Type of prop (determines which stat to aggregate)
        
    Returns:
        Tuple of (updated dataframe, list of new feature names)
        
    Example:
        >>> df, features = add_opponent_defense_features(df, "pass_yds")
        >>> print(features)
        ['opp_pass_yds_allowed_hist']
    """
    print("Adding opponent defense features...")
    
    # Select appropriate target column based on prop type (from config)
    if prop_type not in PROP_TYPE_TO_DEFENSE_TARGET:
        print(f"[INFO] No opponent defense features for {prop_type}")
        return df, []
    
    target_col, feature_name = PROP_TYPE_TO_DEFENSE_TARGET[prop_type]
    
    if target_col not in df.columns:
        print(f"[WARNING] Column {target_col} not found, skipping opponent defense")
        return df, []
    
    # Aggregate yards allowed by opponent per week
    def_agg = (
        df.groupby(['opp_teamid', 'season', 'week'])[target_col]
        .mean()
        .reset_index()
    )
    def_agg = def_agg.rename(columns={target_col: f'opp_{target_col}_allowed'})
    def_agg = def_agg.sort_values(['opp_teamid', 'season', 'week'])
    
    # Calculate historical average (excluding current week to prevent leakage)
    def_agg[feature_name] = (
        def_agg.groupby('opp_teamid', group_keys=False)[f'opp_{target_col}_allowed']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )
    
    # Merge back to main dataframe
    df = df.merge(
        def_agg[['opp_teamid', 'season', 'week', feature_name]],
        on=['opp_teamid', 'season', 'week'],
        how='left'
    )
    
    # Fill missing values (first game of season for defense)
    df[feature_name] = df[feature_name].fillna(0)
    
    print(f"[SUCCESS] Added opponent defense feature: {feature_name}")
    return df, [feature_name]


def add_matchup_history_features(
    df: pd.DataFrame,
    prop_type: PropType
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add player career history vs specific opponent (vectorized).
    
    Calculates player's historical average performance against each specific
    opponent. Uses vectorized operations for optimal performance.
    
    Args:
        df: Player game dataframe with playerid, opp_teamid, season, week
        prop_type: Type of prop (determines which stat to use)
        
    Returns:
        Tuple of (updated dataframe, list of new feature names)
        
    Example:
        >>> df, features = add_matchup_history_features(df, "pass_yds")
        >>> print(features)
        ['player_vs_opp_avg', 'player_vs_opp_games']
    """
    print("Adding matchup history features...")
    
    # Select target column based on prop type (from config)
    target_col = PROP_TYPE_TO_TARGET_COL.get(prop_type)
    
    if not target_col or target_col not in df.columns:
        print(f"[INFO] No matchup history features for {prop_type}")
        return df, []
    
    # Sort for proper time ordering
    df = df.sort_values(['playerid', 'opp_teamid', 'season', 'week'])
    
    # Calculate expanding mean vs opponent (excluding current game to prevent leakage)
    df['player_vs_opp_avg'] = (
        df.groupby(['playerid', 'opp_teamid'])[target_col]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )
    
    # Count of prior games vs opponent
    df['player_vs_opp_games'] = (
        df.groupby(['playerid', 'opp_teamid']).cumcount()
    )
    
    features = ['player_vs_opp_avg', 'player_vs_opp_games']
    print(f"[SUCCESS] Added matchup history features: {features}")
    
    return df, features


def add_advanced_opponent_defense_features(
    df: pd.DataFrame,
    prop_type: PropType,
    start_season: int = 2018
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add advanced opponent defensive metrics from stats.defense_advanced.
    
    Aggregates defensive player stats (pressures, sacks, hits, hurries, blitzes)
    by team and week to create team-level defensive pressure metrics.
    
    Args:
        df: Player game dataframe with opp_teamid, season, week
        prop_type: Type of prop (determines which defensive stats to add)
        start_season: First season to load defensive data from
        
    Returns:
        Tuple of (updated dataframe, list of new feature names)
        
    Example:
        >>> df, features = add_advanced_opponent_defense_features(df, "pass_yds")
        >>> print(features)
        ['opp_def_pressures_per_game', 'opp_def_sacks_per_game', 'opp_def_qb_hits_per_game',
         'opp_def_qb_hurries_per_game', 'opp_def_blitzes_per_game']
    """
    print("Adding advanced opponent defense features...")
    
    required_cols = ['opp_teamid', 'season', 'week']
    if not all(col in df.columns for col in required_cols):
        print(f"[WARNING] Missing required columns for advanced defense features")
        return df, []
    
    # Determine which defensive stats to load based on prop type
    if prop_type == "pass_yds":
        # QB props - load pressure and coverage stats
        defensive_stats = _load_qb_defensive_pressure_stats(start_season)
    else:
        print(f"[INFO] No advanced defensive features for {prop_type}")
        return df, []
    
    if defensive_stats is None or defensive_stats.empty:
        print("[WARNING] No defensive stats loaded")
        return df, []
    
    # Merge defensive stats to player dataframe
    df = df.merge(
        defensive_stats,
        left_on=['opp_teamid', 'season', 'week'],
        right_on=['def_teamid', 'season', 'week'],
        how='left'
    )
    
    # Drop duplicate def_teamid column
    if 'def_teamid' in df.columns:
        df = df.drop(columns=['def_teamid'])
    
    # Get list of new features (excluding merge keys)
    new_features = [col for col in defensive_stats.columns 
                   if col not in ['def_teamid', 'season', 'week']]
    
    # Fill missing values with 0 (first game of season for defense)
    for feat in new_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(0)
    
    print(f"[SUCCESS] Added {len(new_features)} advanced defensive features: {new_features}")
    return df, new_features


def _load_qb_defensive_pressure_stats(start_season: int) -> pd.DataFrame:
    """
    Load QB pressure stats aggregated by defensive team.
    
    Aggregates defensive player pressure stats from stats.defense_advanced
    to get team-level defensive pressure metrics per week.
    Uses expanding window with shift(1) to calculate historical averages.
    
    Args:
        start_season: First season to include
        
    Returns:
        DataFrame with defensive team pressure stats per week
    """
    from datetime import datetime
    current_year = datetime.now().year
    
    # Aggregate defensive player stats by team and week
    query = f"""
    SELECT 
        teamid as def_teamid,
        season,
        week,
        SUM(COALESCE(pressures, 0)) as team_pressures,
        SUM(COALESCE(sacks, 0)) as team_sacks,
        SUM(COALESCE(qb_hits, 0)) as team_qb_hits,
        SUM(COALESCE(qb_hurry, 0)) as team_qb_hurries,
        SUM(COALESCE(blitzes, 0)) as team_blitzes,
        COUNT(DISTINCT playerid) as def_players
    FROM stats.defense_advanced
    WHERE season >= {start_season} 
        AND season <= {current_year}
    GROUP BY teamid, season, week
    ORDER BY teamid, season, week
    """
    
    df = execute_query(query)
    
    if df is None or df.empty:
        print("[WARNING] No QB defensive pressure stats found")
        return pd.DataFrame()
    
    # Calculate historical averages (expanding window, shift(1) to prevent leakage)
    df = df.sort_values(['def_teamid', 'season', 'week'])
    
    # Pressures per game
    df['opp_def_pressures_per_game'] = (
        df.groupby('def_teamid')['team_pressures']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Sacks per game
    df['opp_def_sacks_per_game'] = (
        df.groupby('def_teamid')['team_sacks']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # QB hits per game
    df['opp_def_qb_hits_per_game'] = (
        df.groupby('def_teamid')['team_qb_hits']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # QB hurries per game
    df['opp_def_qb_hurries_per_game'] = (
        df.groupby('def_teamid')['team_qb_hurries']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Blitzes per game
    df['opp_def_blitzes_per_game'] = (
        df.groupby('def_teamid')['team_blitzes']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Keep only necessary columns
    result_cols = ['def_teamid', 'season', 'week', 
                   'opp_def_pressures_per_game', 'opp_def_sacks_per_game', 
                   'opp_def_qb_hits_per_game', 'opp_def_qb_hurries_per_game',
                   'opp_def_blitzes_per_game']
    
    return df[result_cols].fillna(0)


__all__ = [
    'add_opponent_defense_features',
    'add_matchup_history_features',
    'add_advanced_opponent_defense_features'
]