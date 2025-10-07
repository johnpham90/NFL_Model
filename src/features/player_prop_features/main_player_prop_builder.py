"""
Main feature builder for player props.

Orchestrates all feature engineering:
- Base player stats loading
- Position filtering
- Game lines integration
- TD indicators
"""
from typing import Optional
import pandas as pd
from src.config.player_prop_config import PropType, PositionType
from src.features.player_prop_features.player_prop_features import (
    load_player_data as _load_base_data,
    add_td_indicators,
    validate_dataframe
)
from src.features.player_prop_features.player_prop_gamelines_feature import add_game_lines


def build_player_features(
    start_season: int,
    prop_type: PropType,
    position: Optional[PositionType] = None,
    add_indicators: bool = True,
    add_lines: bool = True
) -> pd.DataFrame:
    """
    Main entry point for building player prop features.
    
    Orchestrates the complete feature engineering pipeline:
    1. Load base player stats (passing/rushing/receiving)
    2. Filter by position (if specified)
    3. Add game lines and betting context
    4. Add TD indicators
    
    Args:
        start_season: First season to include
        prop_type: Type of prop ("pass_yds", "rush_yds", "rec_yds", "receptions", "any_td")
        position: Optional position filter (QB, RB, WR, TE, PASS_CATCHER)
        add_indicators: If True, add binary TD indicators
        add_lines: If True, add betting lines and game context
        
    Returns:
        DataFrame with all features engineered and ready for modeling
        
    Example:
        >>> from src.features.player_prop_features.main_player_prop_builder import build_player_features
        >>> df = build_player_features(2020, "pass_yds", position="QB")
        >>> print(df.columns)  # All features included
    """
    print(f"\n{'='*60}")
    print(f"Building features: {prop_type} | Position: {position or 'All'}")
    print(f"{'='*60}\n")
    
    # Step 1: Load base player data with position filtering
    df = _load_base_data(
        start_season=start_season,
        prop_type=prop_type,
        add_indicators=False,  # We'll add indicators at the end
        position=position
    )
    
    # Step 2: Add game lines and betting context
    if add_lines:
        df = add_game_lines(df)
    else:
        print("[INFO] Skipping game lines (add_lines=False)")
    
    # Step 3: Add TD indicators
    if add_indicators:
        df = add_td_indicators(df, prop_type)
    
    # Final validation
    validate_dataframe(df, min_rows=50, context=f"feature builder for {prop_type}")
    
    print(f"\n[SUCCESS] Built {len(df)} records with {len(df.columns)} features")
    print(f"{'='*60}\n")
    
    return df


__all__ = ['build_player_features']