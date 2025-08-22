"""Lean V2 team-level features.

Focused refactor providing a minimal functional API:
  * Parse composite hyphen-delimited stat columns into atomic numeric fields
  * Produce single-game offensive / defensive matrices (week rows x team columns)
  * Produce shifted expanding historical means (no same-game leakage when exclude_current=True)

Public API:
  offensive_metrics(season, stat, exclude_current=True) -> (single_game_df, expanding_mean_df)
  defensive_metrics(season, stat, exclude_current=True) -> (single_game_df, expanding_mean_df)
  list_parsed_stats(season) -> list of available numeric stat column names
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List
import numpy as np
import pandas as pd
from src.utils.db_utils import execute_query
from src.config.config_v2 import team_feature_configs_v2

# Mapping of composite column -> list of parts (legacy names preserved in config)
PARSE_MAP = getattr(team_feature_configs_v2, "parse_data_colums", {}) or {}
LEGACY_TO_CANON = getattr(team_feature_configs_v2, "legacy_to_canonical", {}) or {}
EFF_SPECS = getattr(team_feature_configs_v2, "canonical_efficiency_specs", {}) or {}

def _canon(name: str) -> str:
    """Canonicalize a stat label (spaces -> underscores, lowercase)."""
    return name.strip().lower().replace(" ", "_")

def _parse_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Expand configured composite stat columns into atomic numeric columns.

    Uses legacy->canonical mapping from config to ensure consistent column names
    (e.g. 'fourth down converstions' -> 'fourth_down_conversions').
    """
    for comp_col, parts in PARSE_MAP.items():
        if comp_col not in df.columns:
            continue
        tokens = df[comp_col].fillna("").astype(str).str.split("-", expand=True)
        if tokens.shape[1] < len(parts):
            continue  # malformed row count
        for i, legacy_part in enumerate(parts):
            legacy_key = legacy_part.strip().lower()
            canon_name = LEGACY_TO_CANON.get(legacy_key, _canon(legacy_part))
            df[canon_name] = pd.to_numeric(tokens[i], errors="coerce")
    # Time of possession -> seconds (mm:ss)
    if "time_of_possession" in df.columns:
        top = df["time_of_possession"].fillna("").astype(str)
        mins = pd.to_numeric(top.str.split(":").str[0], errors="coerce")
        secs = pd.to_numeric(top.str.split(":").str[1], errors="coerce")
        df["time_of_possession_seconds"] = (mins * 60 + secs).astype("float32")
    return df

@lru_cache(maxsize=32)
def _load_season(season: int) -> pd.DataFrame:
    """Load and parse raw team stats for one season."""
    q = f"""
    SELECT g.season, g.week, ts.teamid, g.hometeamid, g.awayteamid, ts.*
    FROM stats.teamstats ts
    JOIN stats.gamesummary g ON ts.gamesummaryid = g.gamesummaryid
    WHERE g.season = {season}
    ORDER BY g.season, g.week;
    """
    df = execute_query(q)
    if df is None or df.empty:
        raise ValueError(f"No team stats for season {season}")
    df = df.copy()
    df = _parse_composites(df)
    # Derive efficiency statistics if components available
    for eff_name, (num_col, den_col) in EFF_SPECS.items():
        if num_col in df.columns and den_col in df.columns:
            denom = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
            num = pd.to_numeric(df[num_col], errors="coerce")
            df[eff_name] = (num / denom).astype("float32")
    # Defensive opponent id
    df["def_team"] = np.where(df["teamid"] == df["hometeamid"], df["awayteamid"], df["hometeamid"])
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df

def list_parsed_stats(season: int) -> List[str]:
    """Return list of available numeric stat columns after parsing for a season."""
    df = _load_season(season)
    base = {"season","week","teamid","hometeamid","awayteamid","def_team"}
    numeric = [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric)

def _single_game_matrix(df: pd.DataFrame, stat_col: str, offense: bool) -> pd.DataFrame:
    """Pivot single-game values to (season, week) index x team columns."""
    key_col = "teamid" if offense else "def_team"
    mat = (
        df[["season","week", key_col, stat_col]]
          .pivot_table(index=["season","week"], columns=key_col, values=stat_col, aggfunc="first")
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

def offensive_metrics(season: int, stat: str, exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (single_game_matrix, expanding_mean_matrix) for an offensive stat."""
    df = _load_season(season)
    stat_c = _canon(stat)
    if stat_c not in df.columns:
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not parsed. Available sample: {list_parsed_stats(season)[:25]}"
        )
    single = _single_game_matrix(df, stat_c, offense=True)
    exp = _expanding(single, exclude_current)
    return single, exp

def defensive_metrics(season: int, stat: str, exclude_current: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (single_game_matrix, expanding_mean_matrix) for a defensive stat (opponent values)."""
    df = _load_season(season)
    stat_c = _canon(stat)
    if stat_c not in df.columns:
        raise ValueError(
            f"Stat '{stat}' (canonical '{stat_c}') not parsed. Available sample: {list_parsed_stats(season)[:25]}"
        )
    single = _single_game_matrix(df, stat_c, offense=False)
    exp = _expanding(single, exclude_current)
    return single, exp

__all__ = [
    "offensive_metrics",
    "defensive_metrics",
    "list_parsed_stats",
]