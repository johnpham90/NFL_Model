"""
Main feature builder for player props.

Orchestrates all feature engineering:
- Base player stats loading
- Position filtering
- Game lines integration
- Snap count and usage features
- TD indicators
"""
import logging
from typing import Optional, Tuple, Dict, List
import pandas as pd
from src.config.player_prop_config import PropType, PositionType
from src.features.player_prop_features.player_prop_features import (
    load_player_data as _load_base_data,
    add_td_indicators,
    validate_dataframe
)
from src.features.player_prop_features.player_prop_gamelines_feature import add_game_lines
from src.features.player_prop_features.player_prop_snapcount_feature import (
    add_snap_count_features,
    add_usage_rate_features
)
from src.features.player_prop_features.player_prop_defense_matchup_feature import (
    add_opponent_defense_features,
    add_matchup_history_features,
    add_advanced_opponent_defense_features
)

logger = logging.getLogger(__name__)


def build_player_features(
    start_season: int,
    prop_type: PropType,
    position: Optional[PositionType] = None,
    add_indicators: bool = True,
    add_lines: bool = True,
    add_snap_counts: bool = False,
    include_opponent_defense: bool = True,
    include_advanced_defense: bool = True,
    include_matchup_history: bool = True,
    return_feature_names: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Main entry point for building player prop features.
    
    Orchestrates the complete feature engineering pipeline:
    1. Load base player stats (passing/rushing/receiving)
    2. Filter by position (if specified)
    3. Add game lines and betting context
    4. Add snap count and usage rate features
    5. Add opponent defense features (yards allowed)
    6. Add advanced opponent defense features (pressure, coverage)
    7. Add matchup history features
    8. Add TD indicators
    
    Args:
        start_season: First season to include
        prop_type: Type of prop ("pass_yds", "rush_yds", "rec_yds", "receptions", "any_td")
        position: Optional position filter (QB, RB, WR, TE, PASS_CATCHER)
        add_indicators: If True, add binary TD indicators
        add_lines: If True, add betting lines and game context
        add_snap_counts: If True, add snap count and usage rate features
        include_opponent_defense: If True, add opponent defensive metrics (yards allowed)
        include_advanced_defense: If True, add advanced defensive metrics (pressure, coverage)
        include_matchup_history: If True, add player vs opponent history
        return_feature_names: If True, return (df, feature_dict) instead of just df
        
    Returns:
        If return_feature_names=False: DataFrame with all features
        If return_feature_names=True: (DataFrame, Dict mapping feature category to column names)
    """
    logger.info("="*60)
    logger.info(f"Building features: {prop_type} | Position: {position or 'All'}")
    logger.info("="*60)
    
    feature_dict = {}
    
    # ========== STEP 1: Load Base Player Data ==========
    logger.info("Step 1: Loading base player data...")
    
    df = _load_base_data(
        start_season=start_season,
        prop_type=prop_type,
        add_indicators=False,  # Add at end for proper ordering
        position=position,
        add_lines=False  # Add in step 2 for single source of truth
    )
    
    feature_dict['base'] = list(df.columns)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} base features")
    
    # ========== STEP 2: Add Game Lines ==========
    if add_lines:
        logger.info("Step 2: Adding game lines and betting context...")
        try:
            cols_before = set(df.columns)
            df = add_game_lines(df)
            feature_dict['game_lines'] = list(set(df.columns) - cols_before)
            logger.info(f"Added {len(feature_dict['game_lines'])} game line features")
        except Exception as e:
            logger.warning(f"Failed to add game lines: {e}")
            logger.info("Continuing without game line features")
            feature_dict['game_lines'] = []
    else:
        logger.info("Step 2: Skipping game lines (add_lines=False)")
        feature_dict['game_lines'] = []
    
    # ========== STEP 3: Add Snap Count Features ==========
    if add_snap_counts:
        if position is None:
            logger.warning("Cannot add snap count features without position filter")
            logger.info("Skipping snap count features (position=None)")
            feature_dict['snap_counts'] = []
            feature_dict['usage_rates'] = []
        else:
            logger.info("Step 3: Adding snap count and usage rate features...")
            try:
                df, snap_features = add_snap_count_features(df, position)
                df, usage_features = add_usage_rate_features(df, position)
                
                feature_dict['snap_counts'] = snap_features
                feature_dict['usage_rates'] = usage_features
                
                logger.info(f"Added {len(snap_features)} snap count features")
                logger.info(f"Added {len(usage_features)} usage rate features")
            except Exception as e:
                logger.warning(f"Failed to add snap/usage features: {e}")
                logger.info("Continuing without snap/usage features")
                feature_dict['snap_counts'] = []
                feature_dict['usage_rates'] = []
    else:
        logger.info("Step 3: Skipping snap count features (add_snap_counts=False)")
        feature_dict['snap_counts'] = []
        feature_dict['usage_rates'] = []
    
    # ========== STEP 4: Add Opponent Defense Features ==========
    if include_opponent_defense:
        logger.info("Step 4: Adding opponent defense features...")
        try:
            df, def_features = add_opponent_defense_features(df, prop_type)
            feature_dict['opponent_defense'] = def_features
            logger.info(f"Added {len(def_features)} opponent defense features")
        except Exception as e:
            logger.warning(f"Failed to add opponent defense features: {e}")
            logger.info("Continuing without opponent defense features")
            feature_dict['opponent_defense'] = []
    else:
        logger.info("Step 4: Skipping opponent defense features (include_opponent_defense=False)")
        feature_dict['opponent_defense'] = []
    
    # ========== STEP 5: Add Advanced Opponent Defense Features ==========
    if include_advanced_defense:
        logger.info("Step 5: Adding advanced opponent defense features...")
        try:
            df, adv_def_features = add_advanced_opponent_defense_features(df, prop_type, start_season)
            feature_dict['advanced_defense'] = adv_def_features
            logger.info(f"Added {len(adv_def_features)} advanced defense features")
        except Exception as e:
            logger.warning(f"Failed to add advanced defense features: {e}")
            logger.info("Continuing without advanced defense features")
            feature_dict['advanced_defense'] = []
    else:
        logger.info("Step 5: Skipping advanced defense features (include_advanced_defense=False)")
        feature_dict['advanced_defense'] = []
    
    # ========== STEP 6: Add Matchup History Features ==========
    if include_matchup_history:
        logger.info("Step 6: Adding matchup history features...")
        try:
            df, matchup_features = add_matchup_history_features(df, prop_type)
            feature_dict['matchup_history'] = matchup_features
            logger.info(f"Added {len(matchup_features)} matchup history features")
        except Exception as e:
            logger.warning(f"Failed to add matchup history features: {e}")
            logger.info("Continuing without matchup history features")
            feature_dict['matchup_history'] = []
    else:
        logger.info("Step 6: Skipping matchup history features (include_matchup_history=False)")
        feature_dict['matchup_history'] = []
    
    # ========== STEP 7: Add TD Indicators ==========
    if add_indicators:
        logger.info("Step 7: Adding TD indicators...")
        try:
            cols_before = set(df.columns)
            df = add_td_indicators(df, prop_type)
            feature_dict['indicators'] = list(set(df.columns) - cols_before)
            logger.info(f"Added {len(feature_dict['indicators'])} TD indicator features")
        except Exception as e:
            logger.warning(f"Failed to add TD indicators: {e}")
            logger.info("Continuing without TD indicators")
            feature_dict['indicators'] = []
    else:
        logger.info("Step 7: Skipping TD indicators (add_indicators=False)")
        feature_dict['indicators'] = []
    
    # ========== VALIDATION ==========
    validate_dataframe(df, min_rows=50, context=f"feature builder for {prop_type}")
    
    # ========== SUMMARY ==========
    _print_feature_summary(df, feature_dict, prop_type, position)
    
    if return_feature_names:
        return df, feature_dict
    return df


def _print_feature_summary(
    df: pd.DataFrame, 
    feature_dict: Dict[str, List[str]], 
    prop_type: str, 
    position: Optional[str]
) -> None:
    """Print summary of feature engineering results."""
    logger.info("")
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("="*60)
    logger.info(f"Prop Type: {prop_type}")
    logger.info(f"Position: {position or 'All'}")
    logger.info(f"Total Records: {len(df):,}")
    logger.info(f"Total Features: {len(df.columns)}")
    
    if 'season' in df.columns:
        logger.info(f"Seasons: {df['season'].min()}-{df['season'].max()}")
    if 'week' in df.columns:
        logger.info(f"Weeks: {df['week'].min()}-{df['week'].max()}")
    if 'playerid' in df.columns:
        logger.info(f"Unique Players: {df['playerid'].nunique():,}")
    
    logger.info("")
    logger.info("Features by Category:")
    for category, features in feature_dict.items():
        logger.info(f"  {category}: {len(features)} features")
    
    logger.info("="*60)
    logger.info("")


__all__ = ['build_player_features']