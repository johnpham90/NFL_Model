import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query
from src.config.config_v2 import player_feature_configs_v2

"""Player-level features module.

Provides a minimal functional API:
  * Aggregates player-level offense and defense stats to team-game level (only players who played)
  * Produces single-game offensive/defensive matrices (week rows x team columns)
  * Produces shifted expanding historical means (no same-game leakage when exclude_current=True)

Public API:
  offensive_player_metrics(season, stat, exclude_current=True) -> (single_game_df, expanding_mean_df)
  defensive_player_metrics(season, stat, exclude_current=True) -> (single_game_df, expanding_mean_df)
  list_parsed_player_stats(season) -> list of available numeric stat column names
"""
from functools import lru_cache
from typing import Tuple, List
import numpy as np
import pandas as pd
from src.utils.db_utils import execute_query

# --- CONFIG ---

OFFENSE_STAT_COLS = player_feature_configs_v2.offense_stat_cols
DEFENSE_STAT_COLS = player_feature_configs_v2.defense_stat_cols

def _canon(name: str) -> str:
    """Canonicalize a stat label (spaces -> underscores, lowercase)."""
    return name.strip().lower().replace(" ", "_")

# --- DATA LOADING ---

@lru_cache(maxsize=16)
def _load_player_stats(season: int, table: str) -> pd.DataFrame:
    """Load player-level stats for one season from the given table."""
    q = f"""
    SELECT *
    FROM {table}
    WHERE season = {season}
    ORDER BY season, week, teamid, playerid;
    """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No player stats for {table} in season {season}")
    return df.copy()

@lru_cache(maxsize=16)
def _load_snapcounts(season: int) -> pd.DataFrame:
    """Load snapcounts for one season."""
    q = f"""
    SELECT season, week, teamid, playerid
    FROM stats.snapcounts
    WHERE season = {season}
    """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No snapcounts for season {season}")
    return df.copy()

def _aggregate_team_game_player_stats(player_stats_df: pd.DataFrame, snapcounts_df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """Aggregate player-level stats to team-game level, only for players who played."""
    merged = pd.merge(
        player_stats_df,
        snapcounts_df[['season', 'week', 'teamid', 'playerid']],
        on=['season', 'week', 'teamid', 'playerid'],
        how='inner'
    )
    agg = merged.groupby(['season', 'week', 'teamid'])[stat_cols].sum().reset_index()
    return agg

def _single_game_matrix(df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """Pivot single-game values to (season, week) index x team columns."""
    mat = (
        df[["season", "week", "teamid", stat_col]]
          .pivot_table(index=["season", "week"], columns="teamid", values=stat_col, aggfunc="first")
          .sort_index()
          .astype("float32")
    )
    return mat

def _expanding(mat: pd.DataFrame, exclude_current: bool) -> pd.DataFrame:
    """Season-wise expanding mean (optionally shifted to exclude current game)."""
    out = []
    for season, block in mat.groupby(level="season"):
        exp = block.expanding(min_periods=1).mean()
        if exclude_current:
            exp = exp.groupby(level="season").shift(1)
        out.append(exp)
    return pd.concat(out).sort_index().astype("float32")

def list_parsed_player_stats(season: int, offense: bool = True) -> List[str]:
    """Return list of available numeric stat columns for player-level features for a season."""
    table = "stats.offense" if offense else "stats.defense"
    df = _load_player_stats(season, table)
    base = {"season", "week", "teamid", "playerid"}
    numeric = [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric)

def offensive_player_metrics(season: int, stat: str, exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (single_game_matrix, expanding_mean_matrix) for an offensive player stat."""
    df = _load_player_stats(season, "stats.offense")
    snap = _load_snapcounts(season)
    stat_c = _canon(stat)
    if stat_c not in df.columns:
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not found. Available: {list_parsed_player_stats(season, offense=True)}"
        )
    agg = _aggregate_team_game_player_stats(df, snap, [stat_c])
    single = _single_game_matrix(agg, stat_c)
    exp = _expanding(single, exclude_current)
    return single, exp

def defensive_player_metrics(season: int, stat: str, exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (single_game_matrix, expanding_mean_matrix) for a defensive player stat."""
    df = _load_player_stats(season, "stats.defense")
    snap = _load_snapcounts(season)
    stat_c = _canon(stat)
    if stat_c not in df.columns:
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not found. Available: {list_parsed_player_stats(season, offense=False)}"
        )
    agg = _aggregate_team_game_player_stats(df, snap, [stat_c])
    single = _single_game_matrix(agg, stat_c)
    exp = _expanding(single, exclude_current)
    return single, exp

__all__ = [
    "offensive_player_metrics",
    "defensive_player_metrics",
    "list_parsed_player_stats",
]