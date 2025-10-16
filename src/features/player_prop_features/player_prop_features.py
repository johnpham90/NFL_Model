"""Player prop feature engineering utilities.

Focused refactor providing minimal functional API for player-level statistics:
  * Load and parse player stats from passing, rushing, receiving tables
  * Join with offense table for TD data
  * Produce single-game player matrices (week rows x player columns)
  * Produce shifted expanding historical means (no same-game leakage when exclude_current=True)
  * Calculate efficiency metrics (catch rate, yards per attempt, etc.)

Public API:
  player_offensive_metrics(season, player_id, stat, prop_type, exclude_current=True) 
      -> (single_game_df, expanding_mean_df)
  player_opponent_defense_metrics(season, team_id, stat, prop_type, exclude_current=True)
      -> (single_game_df, expanding_mean_df)
  list_player_stats(season, prop_type) -> list of available stat column names
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List, Literal
import numpy as np
import pandas as pd
from src.utils.db_utils import execute_query
from src.config.player_prop_config import player_prop_feature_config, PositionType

PropType = Literal["pass_yds", "rush_yds", "rec_yds", "receptions", "any_td"]

# Efficiency calculation specs from config
PASSING_EFF_SPECS = player_prop_feature_config.passing_efficiency_specs
RUSHING_EFF_SPECS = player_prop_feature_config.rushing_efficiency_specs
RECEIVING_EFF_SPECS = player_prop_feature_config.receiving_efficiency_specs


# ========== UTILITY FUNCTIONS ==========

def _canon(name: str) -> str:
    """Canonicalize a stat label (lowercase, underscores)."""
    return name.strip().lower().replace(" ", "_")


def add_opponent_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent team ID based on home/away status.
    
    Args:
        df: DataFrame with teamid, hometeamid, awayteamid columns
        
    Returns:
        DataFrame with added opp_teamid column
    """
    df['opp_teamid'] = np.where(
        df['teamid'] == df['hometeamid'], 
        df['awayteamid'], 
        df['hometeamid']
    )
    return df


def add_home_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary home game indicator.
    
    Args:
        df: DataFrame with teamid and hometeamid columns
        
    Returns:
        DataFrame with added is_home column
    """
    df['is_home'] = (df['teamid'] == df['hometeamid']).astype(int)
    return df


def add_td_indicators(df: pd.DataFrame, prop_type: PropType) -> pd.DataFrame:
    """
    Add binary TD indicator columns based on prop type.
    
    Args:
        df: DataFrame with TD columns
        prop_type: Type of prop (determines which TD indicator to add)
        
    Returns:
        DataFrame with added has_*_td column(s)
    """
    if prop_type == "pass_yds" and 'pass_td' in df.columns:
        df['has_pass_td'] = (df['pass_td'] > 0).astype(int)
    elif prop_type == "rush_yds" and 'rush_td' in df.columns:
        df['has_rush_td'] = (df['rush_td'] > 0).astype(int)
    elif prop_type in ["rec_yds", "receptions"] and 'rec_td' in df.columns:
        df['has_rec_td'] = (df['rec_td'] > 0).astype(int)
    elif prop_type == "any_td":  # ADD THIS
        if 'has_any_td' not in df.columns:
            df['has_any_td'] = ((df.get('rush_td', 0) > 0) | (df.get('rec_td', 0) > 0)).astype(int)
    return df


def validate_dataframe(df: pd.DataFrame, min_rows: int = 50, context: str = "") -> None:
    """
    Validate DataFrame has sufficient data.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        context: Optional context string for error message
        
    Raises:
        ValueError: If DataFrame is empty or below minimum
    """
    if df is None or df.empty:
        raise ValueError(f"DataFrame is empty or None{' (' + context + ')' if context else ''}")
    
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient data{' for ' + context if context else ''}: "
            f"{len(df)} rows (minimum {min_rows} required)"
        )


def validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """
    Validate target column exists in DataFrame.
    
    Args:
        df: DataFrame to check
        target_col: Target column name
        
    Raises:
        ValueError: If target column not found
    """
    if target_col not in df.columns:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"Target column '{target_col}' not found in dataset.\n"
            f"Available columns ({len(available)}): {available[:20]}..."
        )


def validate_required_columns(df: pd.DataFrame, required_cols: list, context: str = "") -> None:
    """
    Validate all required columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        context: Optional context string for error message
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns{' for ' + context if context else ''}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def add_position_features(df: pd.DataFrame, prop_type: PropType) -> pd.DataFrame:
    """
    Add position indicators and position-specific interaction features.
    
    For receiving props, creates features that capture position-specific patterns:
    - RB: More checkdowns, screens, higher catch rate, lower yards per target
    - WR: Deep routes, more air yards, boom/bust profiles
    - TE: Red zone targets, intermediate routes, first downs
    
    Args:
        df: DataFrame with current_position column
        prop_type: Type of prop being predicted
        
    Returns:
        DataFrame with added position features and interactions
    """
    if 'current_position' not in df.columns:
        print("[WARNING] No current_position column - skipping position features")
        return df
    
    # Basic one-hot encoding
    df['is_rb'] = (df['current_position'] == 'RB').astype(int)
    df['is_wr'] = (df['current_position'] == 'WR').astype(int)
    df['is_te'] = (df['current_position'] == 'TE').astype(int)
    
    features_added = ['is_rb', 'is_wr', 'is_te']
    
    # Position-specific interaction features for receiving props
    if prop_type in ['rec_yds', 'receptions']:
        
        # RB-specific features (short routes, high catch rate)
        if 'targets' in df.columns:
            df['rb_target_vol'] = df['is_rb'] * df['targets']
            features_added.append('rb_target_vol')
        
        if 'rec_adot' in df.columns:
            df['rb_shallow_routes'] = df['is_rb'] * (df['rec_adot'] < 5).astype(int)
            features_added.append('rb_shallow_routes')
        
        if 'rec' in df.columns and 'targets' in df.columns:
            # Catch rate for RBs (typically higher than WR/TE)
            catch_rate = np.where(df['targets'] > 0, df['rec'] / df['targets'], 0)
            df['rb_catch_efficiency'] = df['is_rb'] * catch_rate
            features_added.append('rb_catch_efficiency')
        
        # WR-specific features (deep routes, air yards)
        if 'rec_air_yds' in df.columns:
            df['wr_air_yards'] = df['is_wr'] * df['rec_air_yds']
            features_added.append('wr_air_yards')
        
        if 'rec_adot' in df.columns:
            df['wr_deep_routes'] = df['is_wr'] * (df['rec_adot'] > 10).astype(int)
            features_added.append('wr_deep_routes')
        
        if 'rec_yds' in df.columns and 'targets' in df.columns:
            # Yards per target for WRs (typically higher than RB/TE)
            yds_per_tgt = np.where(df['targets'] > 0, df['rec_yds'] / df['targets'], 0)
            df['wr_explosiveness'] = df['is_wr'] * yds_per_tgt
            features_added.append('wr_explosiveness')
        
        # TE-specific features (red zone, intermediate, first downs)
        if 'rec_first_down' in df.columns:
            df['te_first_downs'] = df['is_te'] * df['rec_first_down']
            features_added.append('te_first_downs')
        
        if 'rec_adot' in df.columns:
            # TEs typically work intermediate (5-15 yard) routes
            is_intermediate = df['rec_adot'].between(5, 15).astype(int)
            df['te_intermediate'] = df['is_te'] * is_intermediate
            features_added.append('te_intermediate')
        
        if 'rec_yac' in df.columns:
            # TEs often good YAC in intermediate game
            df['te_yac'] = df['is_te'] * df['rec_yac']
            features_added.append('te_yac')
    
    print(f"[INFO] Added {len(features_added)} position features for {prop_type}")
    return df


# ========== EFFICIENCY CALCULATIONS ==========


def _calculate_efficiencies(df: pd.DataFrame, prop_type: PropType) -> pd.DataFrame:
    """Calculate efficiency metrics based on prop type."""
    if prop_type == "pass_yds":
        for eff_name, (num_col, den_col) in PASSING_EFF_SPECS.items():
            if num_col in df.columns and den_col in df.columns:
                denom = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
                num = pd.to_numeric(df[num_col], errors="coerce")
                df[eff_name] = (num / denom).astype("float32")
    
    elif prop_type == "rush_yds":
        for eff_name, (num_col, den_col) in RUSHING_EFF_SPECS.items():
            if num_col in df.columns and den_col in df.columns:
                denom = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
                num = pd.to_numeric(df[num_col], errors="coerce")
                df[eff_name] = (num / denom).astype("float32")
    
    elif prop_type in ["rec_yds", "receptions"]:
        for eff_name, (num_col, den_col) in RECEIVING_EFF_SPECS.items():
            if num_col in df.columns and den_col in df.columns:
                denom = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
                num = pd.to_numeric(df[num_col], errors="coerce")
                df[eff_name] = (num / denom).astype("float32")
    
    return df


@lru_cache(maxsize=32)
def _load_passing_season(season: int) -> pd.DataFrame:
    """Load and parse passing stats for one season with TDs from offense table and QB rushing stats."""
    q = f"""
    SELECT 
        p.gamesummaryid,
        p.playerid,
        p.player,
        p.teamid,
        p.season,
        p.week,
        p.hometeamid,
        p.awayteamid,
        p.pass_att,
        p.pass_cmp,
        p.pass_yds,
        COALESCE(p.pass_air_yds, 0) as pass_air_yds,
        COALESCE(p.pass_air_yds_per_att, 0) as pass_air_yds_per_att,
        COALESCE(p.pass_air_yds_per_cmp, 0) as pass_air_yds_per_cmp,
        COALESCE(p.pass_yac, 0) as pass_yac,
        COALESCE(p.pass_yac_per_cmp, 0) as pass_yac_per_cmp,
        COALESCE(p.pass_target_yds, 0) as pass_target_yds,
        COALESCE(p.pass_tgt_yds_per_att, 0) as pass_tgt_yds_per_att,
        p.pass_poor_throw_pct,
        p.pass_drop_pct,
        COALESCE(p.pass_first_down, 0) as pass_first_down,
        COALESCE(p.pass_first_down_pct, 0) as pass_first_down_pct,
        COALESCE(p.pass_pressured, 0) as pass_pressured,
        p.pass_pressured_pct,
        COALESCE(p.pass_sacked, 0) as pass_sacked,
        COALESCE(p.pass_blitzed, 0) as pass_blitzed,
        COALESCE(p.pass_hits, 0) as pass_hits,
        COALESCE(p.pass_hurried, 0) as pass_hurried,
        COALESCE(o.pass_td, 0) as pass_td,
        COALESCE(o.pass_int, 0) as pass_int,
        COALESCE(o.pass_rating, 0) as pass_rating,
        -- QB rushing stats as features
        COALESCE(r.rush_att, 0) as qb_rush_att,
        COALESCE(r.rush_yds, 0) as qb_rush_yds,
        COALESCE(r.rush_td, 0) as qb_rush_td,
        COALESCE(r.rush_yac, 0) as qb_rush_yac,
        COALESCE(r.rush_broken_tackles, 0) as qb_rush_broken_tackles,
        -- Position info
        pl.current_position,
        CASE WHEN p.teamid = p.hometeamid THEN 1 ELSE 0 END as is_home
        FROM stats.passing p
        LEFT JOIN stats.offense o 
            ON p.gamesummaryid = o.gamesummaryid 
            AND p.playerid = o.playerid
        LEFT JOIN stats.rushing r
            ON p.gamesummaryid = r.gamesummaryid
            AND p.playerid = r.playerid
        LEFT JOIN stats.player pl
            ON p.playerid = pl.playerid
        WHERE p.season = {season}
            AND p.pass_att > 0
        ORDER BY p.season, p.week, p.playerid
        """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No passing stats for season {season}")
    
    df = df.copy()
    
    # Convert percentage text columns to floats
    pct_columns = ['pass_poor_throw_pct', 'pass_drop_pct', 'pass_pressured_pct']
    for col in pct_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna('0%')
                .str.replace('%', '', regex=False)
                .astype(float) / 100.0
            )
    
    # Add opponent team and calculate efficiency metrics
    df = add_opponent_team(df)
    df = _calculate_efficiencies(df, "pass_yds")
    
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df


@lru_cache(maxsize=32)
def _load_rushing_season(season: int) -> pd.DataFrame:
    """Load and parse rushing stats for one season."""
    q = f"""
    SELECT 
        r.gamesummaryid,
        r.playerid,
        r.player,
        r.teamid,
        r.season,
        r.week,
        r.hometeamid,
        r.awayteamid,
        r.rush_att,
        r.rush_yds,
        r.rush_td,
        COALESCE(r.rush_yac, 0) as rush_yac,
        COALESCE(r.rush_yac_per_rush, 0) as rush_yac_per_rush,
        COALESCE(r.rush_yds_before_contact, 0) as rush_yds_before_contact,
        COALESCE(r.rush_yds_bc_per_rush, 0) as rush_yds_bc_per_rush,
        COALESCE(r.rush_broken_tackles, 0) as rush_broken_tackles,
        COALESCE(r.rush_broken_tackles_per_rush, 0) as rush_broken_tackles_per_rush,
        COALESCE(r.rush_first_down, 0) as rush_first_down,
        COALESCE(o.fumbles, 0) as fumbles,
        COALESCE(o.fumbles_lost, 0) as fumbles_lost,
        p.current_position,
        CASE WHEN r.teamid = r.hometeamid THEN 1 ELSE 0 END as is_home
    FROM stats.rushing r
    LEFT JOIN stats.offense o 
        ON r.gamesummaryid = o.gamesummaryid 
        AND r.playerid = o.playerid
    LEFT JOIN stats.player p
        ON r.playerid = p.playerid
    WHERE r.season = {season}
        AND r.rush_att > 0
    ORDER BY r.season, r.week, r.playerid
    """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No rushing stats for season {season}")
    
    df = df.copy()
    
    # Add opponent team and calculate efficiency metrics
    df = add_opponent_team(df)
    df = _calculate_efficiencies(df, "rush_yds")
    
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df


@lru_cache(maxsize=32)
def _load_receiving_season(season: int) -> pd.DataFrame:
    """Load and parse receiving stats for one season."""
    q = f"""
    SELECT 
        r.gamesummaryid,
        r.playerid,
        r.player,
        r.teamid,
        r.season,
        r.week,
        r.hometeamid,
        r.awayteamid,
        r.targets,
        r.rec,
        r.rec_yds,
        r.rec_td,
        COALESCE(r.rec_adot, 0) as rec_adot,
        COALESCE(r.rec_air_yds, 0) as rec_air_yds,
        COALESCE(r.rec_air_yds_per_rec, 0) as rec_air_yds_per_rec,
        COALESCE(r.rec_yac, 0) as rec_yac,
        COALESCE(r.rec_yac_per_rec, 0) as rec_yac_per_rec,
        COALESCE(r.rec_drop_pct, 0) as rec_drop_pct,
        COALESCE(r.rec_drops, 0) as rec_drops,
        COALESCE(r.rec_broken_tackles, 0) as rec_broken_tackles,
        COALESCE(r.rec_broken_tackles_per_rec, 0) as rec_broken_tackles_per_rec,
        COALESCE(r.rec_first_down, 0) as rec_first_down,
        COALESCE(r.rec_pass_rating, 0) as rec_pass_rating,
        COALESCE(r.rec_target_int, 0) as rec_target_int,
        p.current_position,
        CASE WHEN r.teamid = r.hometeamid THEN 1 ELSE 0 END as is_home
    FROM stats.receiving r
    LEFT JOIN stats.offense o 
        ON r.gamesummaryid = o.gamesummaryid 
        AND r.playerid = o.playerid
    LEFT JOIN stats.player p
        ON r.playerid = p.playerid
    WHERE r.season = {season}
        AND r.targets > 0
    ORDER BY r.season, r.week, r.playerid
    """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No receiving stats for season {season}")
    
    df = df.copy()
    
    # Add opponent team and calculate efficiency metrics
    df = add_opponent_team(df)
    df = _calculate_efficiencies(df, "rec_yds")
    
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df


def _load_season(season: int, prop_type: PropType) -> pd.DataFrame:
    """Load appropriate data based on prop type."""
    if prop_type == "pass_yds":
        return _load_passing_season(season)
    elif prop_type == "rush_yds":
        return _load_rushing_season(season)
    elif prop_type in ["rec_yds", "receptions"]:
        return _load_receiving_season(season)
    else:
        raise ValueError(f"Unknown prop_type: {prop_type}")


def list_player_stats(season: int, prop_type: PropType) -> List[str]:
    """Return list of available numeric stat columns for a prop type."""
    df = _load_season(season, prop_type)
    base = {"season", "week", "playerid", "player", "teamid", "gamesummaryid",
            "hometeamid", "awayteamid", "opp_teamid", "is_home"}
    numeric = [c for c in df.columns 
               if c not in base and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric)


def _single_game_player_matrix(df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """Pivot single-game values to (season, week) index x player columns."""
    mat = (
        df[["season", "week", "playerid", stat_col]]
          .pivot_table(index=["season", "week"], columns="playerid", 
                      values=stat_col, aggfunc="first")
          .sort_index()
          .astype("float32")
    )
    return mat


def _expanding_by_player(mat: pd.DataFrame, exclude_current: bool) -> pd.DataFrame:
    """Player-wise expanding mean (optionally shifted to exclude current game)."""
    # For each player column, calculate expanding mean
    result = pd.DataFrame(index=mat.index)
    
    for player_col in mat.columns:
        player_series = mat[player_col]
        exp = player_series.expanding(min_periods=1).mean()
        
        if exclude_current:
            exp = exp.shift(1)
        
        result[player_col] = exp
    
    return result.sort_index().astype("float32")


def player_offensive_metrics(season: int, 
                            stat: str, 
                            prop_type: PropType,
                            exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (single_game_matrix, expanding_mean_matrix) for a player offensive stat.
    
    Args:
        season: NFL season year
        stat: Stat column name (e.g., 'pass_yds', 'rush_yds', 'rec_yds')
        prop_type: Type of prop (pass_yds, rush_yds, rec_yds, receptions)
        exclude_current: If True, shift expanding mean to exclude current game
        
    Returns:
        Tuple of (single_game_df, expanding_mean_df)
    """
    df = _load_season(season, prop_type)
    stat_c = _canon(stat)
    
    if stat_c not in df.columns:
        available = list_player_stats(season, prop_type)
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not found. "
            f"Available stats: {available[:25]}"
        )
    
    single = _single_game_player_matrix(df, stat_c)
    exp = _expanding_by_player(single, exclude_current)
    
    return single, exp


def player_opponent_defense_metrics(season: int,
                                   stat: str,
                                   prop_type: PropType,
                                   exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return opponent defensive stats (yards allowed) aggregated by defensive team.
    
    Args:
        season: NFL season year
        stat: Stat to aggregate (e.g., 'pass_yds', 'rush_yds', 'rec_yds')
        prop_type: Type of prop
        exclude_current: If True, shift to exclude current game
        
    Returns:
        Tuple of (single_game_df, expanding_mean_df) by defensive team
    """
    df = _load_season(season, prop_type)
    stat_c = _canon(stat)
    
    if stat_c not in df.columns:
        available = list_player_stats(season, prop_type)
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not found. "
            f"Available stats: {available[:25]}"
        )
    
    # Aggregate by opponent (defensive team) - yards allowed
    def_mat = (
        df[["season", "week", "opp_teamid", stat_c]]
          .groupby(["season", "week", "opp_teamid"])[stat_c]
          .mean()
          .reset_index()
          .pivot_table(index=["season", "week"], columns="opp_teamid",
                      values=stat_c, aggfunc="first")
          .sort_index()
          .astype("float32")
    )
    
    # Expanding mean by defensive team
    result = pd.DataFrame(index=def_mat.index)
    for team_col in def_mat.columns:
        team_series = def_mat[team_col]
        exp = team_series.expanding(min_periods=1).mean()
        
        if exclude_current:
            exp = exp.shift(1)
        
        result[team_col] = exp
    
    return def_mat, result.sort_index().astype("float32")


# ========== UNIFIED DATA LOADER ==========

def load_player_data(start_season: int, prop_type: PropType, add_indicators: bool = True, position: PositionType = None, add_lines: bool = False) -> pd.DataFrame:
    """
    Unified data loader for all prop types across multiple seasons.
    
    Consolidates loading logic to eliminate duplication between model and feature modules.
    Automatically applies:
    - Opponent team calculation
    - Efficiency metrics (from config)
    - Position filtering (optional)
    - Position features and interactions (for receiving props)
    - Game lines (optional)
    - Optional TD indicators
    
    Args:
        start_season: First season to include
        prop_type: Type of prop data to load ("pass_yds", "rush_yds", "rec_yds", "receptions", "any_td")
        add_indicators: If True, add binary TD indicators
        position: Optional position filter (QB, RB, WR, TE, PASS_CATCHER). 
                  Use PASS_CATCHER to get all pass catchers (RB+WR+TE combined)
        add_lines: If True, add betting lines and game context
        
    Returns:
        DataFrame with player game data, efficiencies calculated, and indicators added
        
    Raises:
        ValueError: If prop_type is unknown or no data returned
    """
    # Special case for any_td - use combined rushing + receiving data
    if prop_type == "any_td":
        df = load_any_td_data(start_season, position)
        if add_lines:
            from src.features.player_prop_features.player_prop_gamelines_feature import add_game_lines
            df = add_game_lines(df)
        return df
    
    # Determine which seasons to load
    from datetime import datetime
    current_year = datetime.now().year
    seasons = range(start_season, current_year + 1)
    
    all_data = []
    for season in seasons:
        try:
            season_df = _load_season(season, prop_type)
            all_data.append(season_df)
        except ValueError as e:
            # Skip seasons with no data
            print(f"[INFO] Skipping season {season}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No data found for prop_type '{prop_type}' from season {start_season} onwards")
    
    # Combine all seasons
    df = pd.concat(all_data, ignore_index=True)
    
    # POSITION FILTERING - only filter if position is explicitly specified
    if position and 'current_position' in df.columns:
        original_count = len(df)
        
        if position == "RB":
            df = df[df['current_position'] == 'RB']
            print(f"[INFO] Filtered to RBs: {original_count} -> {len(df)} records")
        elif position == "PASS_CATCHER":
            # Include RB, WR, TE - everyone who catches passes
            df = df[df['current_position'].isin(['RB', 'WR', 'TE'])]
            print(f"[INFO] Kept all pass catchers (RB/WR/TE): {original_count} -> {len(df)} records")
        elif position == "QB":
            df = df[df['current_position'] == 'QB']
            print(f"[INFO] Filtered to QBs: {original_count} -> {len(df)} records")
        elif position in ["WR", "TE"]:
            df = df[df['current_position'] == position]
            print(f"[INFO] Filtered to {position}: {original_count} -> {len(df)} records")
    elif prop_type in ['rec_yds', 'receptions'] and 'current_position' in df.columns:
        # If no position specified for receiving props, keep all pass catchers
        original_count = len(df)
        df = df[df['current_position'].isin(['RB', 'WR', 'TE'])]
        print(f"[INFO] Kept all pass catchers by default: {original_count} -> {len(df)} records")
    
    # ADD POSITION FEATURES for receiving props (creates interaction features)
    if prop_type in ['rec_yds', 'receptions']:
        df = add_position_features(df, prop_type)
    
    # ADD GAME LINES if requested
    if add_lines:
        from src.features.player_prop_features.player_prop_gamelines_feature import add_game_lines
        df = add_game_lines(df)
    
    # Validate
    validate_dataframe(df, min_rows=50, context=f"{prop_type} from season {start_season}")
    
    # Add TD indicators if requested
    if add_indicators:
        df = add_td_indicators(df, prop_type)
    
    print(f"[INFO] Loaded {len(df)} player-game records for {prop_type} from season {start_season}")
    return df

def load_rb_combined_data(start_season: int) -> pd.DataFrame:
    """Load rushing + receiving data for RBs to predict any TD."""
    rushing = load_player_data(start_season, "rush_yds", add_indicators=True)
    receiving = load_player_data(start_season, "rec_yds", add_indicators=True)
    
    # Merge on player-game level - include more receiving features
    combined = rushing.merge(
        receiving[['gamesummaryid', 'playerid', 'has_rec_td', 'targets', 'rec', 'rec_yds',
                   'rec_adot', 'rec_air_yds', 'rec_yac', 'rec_yac_per_rec', 
                   'rec_broken_tackles_per_rec', 'catch_rate', 'yds_per_target', 
                   'yds_per_rec', 'td_rate']],
        on=['gamesummaryid', 'playerid'],
        how='outer'
    ).fillna(0)
    
    # Create combined TD indicator
    combined['has_any_td'] = ((combined['has_rush_td'] > 0) | (combined['has_rec_td'] > 0)).astype(int)
    
    print(f"[INFO] Loaded {len(combined)} combined RB records for any_td from season {start_season}")
    return combined

def load_any_td_data(start_season: int, position: PositionType = None) -> pd.DataFrame:
    """
    Load position-appropriate data for anytime TD prediction.
    
    Note: 
    - RB: Combines rushing + receiving (can score both ways)
    - WR/TE: Only receiving (rarely rush)
    - PASS_CATCHER: Not recommended for any_td (mixed RB/WR/TE have different TD patterns)
    """
    if position == "RB":
        # RBs can score rushing OR receiving TDs - need both datasets
        return load_rb_combined_data(start_season)
    elif position in ["WR", "TE"]:
        # WR/TE only score receiving TDs (rushing TDs extremely rare)
        df = load_player_data(start_season, "rec_yds", add_indicators=True, position=position)
        df['has_any_td'] = df.get('has_rec_td', 0)
        return df
    elif position == "PASS_CATCHER":
        # PASS_CATCHER includes RBs who can score rushing TDs - need special handling
        raise NotImplementedError(
            "PASS_CATCHER any_td not supported. RBs can score rushing TDs, "
            "but WR/TE cannot. Use position='RB' for RB any_td predictions, "
            "or use rec_yds TD probability for pure receiving TDs."
        )
    else:
        raise ValueError(f"Position {position} not supported for any_td")
__all__ = [
    "player_offensive_metrics",
    "player_opponent_defense_metrics",
    "list_player_stats",
    "load_player_data",
    "load_any_td_data",  
    "add_opponent_team",
    "add_home_indicator",
    "add_td_indicators",
    "validate_dataframe",
    "validate_target_column",
    "validate_required_columns",
]