from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List
import numpy as np
import pandas as pd
from src.utils.db_utils import execute_query
from src.config.config_v2 import player_feature_configs_v2

"""Player level features implementation for NFLModelV2 integration.

Add this method to your NFLModelV2 class to integrate player features.
"""

def build_player_feature_matrices(self, exclude_current: bool = True, 
                                   include_defense: bool = True, 
                                   include_differentials: bool = True):
    """
    Build player-level rolling average features with snap count and bye week awareness.
    Only processes stats relevant to each position to avoid noise features.
    """
    if not hasattr(self, "_hist_df"):
        raise RuntimeError("Call build_feature_matrices first to create team features.")
    
    # Get unique seasons from existing games
    seasons = self._hist_df['season'].unique()
    
    # Load bye week schedule
    bye_schedule = self._load_bye_week_schedule(seasons)
    
    # Load player data for these seasons with snap counts
    all_player_data = []
    for season in seasons:
        q = f"""
        SELECT 
            g.season, g.week, o.playerid, o.player, 
            p.current_position as position, o.teamid, o.gamesummaryid,
            g.hometeamid, g.awayteamid,
            COALESCE(sc.off_num, 0) as offensive_snaps,
            COALESCE(o.pass_yds, 0) as pass_yds,
            COALESCE(o.pass_td, 0) as pass_td,
            COALESCE(o.pass_int, 0) as pass_int,
            COALESCE(o.pass_cmp, 0) as pass_cmp,
            COALESCE(o.pass_att, 0) as pass_att,
            o.pass_rating,
            COALESCE(o.pass_long, 0) as pass_long,
            COALESCE(o.pass_sacked, 0) as pass_sacked,
            COALESCE(o.pass_sacked_yds, 0) as pass_sacked_yds,
            COALESCE(o.rush_yds, 0) as rush_yds,
            COALESCE(o.rush_td, 0) as rush_td,
            COALESCE(o.rush_att, 0) as rush_att,
            COALESCE(o.rush_long, 0) as rush_long,
            COALESCE(o.rec_yds, 0) as rec_yds,
            COALESCE(o.rec_td, 0) as rec_td,
            COALESCE(o.rec, 0) as rec,
            COALESCE(o.targets, 0) as targets,
            COALESCE(o.rec_long, 0) as rec_long,
            COALESCE(o.fumbles, 0) as fumbles,
            COALESCE(o.fumbles_lost, 0) as fumbles_lost
        FROM stats.offense o
        JOIN stats.gamesummary g ON o.gamesummaryid = g.gamesummaryid
        JOIN stats.player p ON o.playerid = p.playerid
        LEFT JOIN stats.snapcounts sc ON o.playerid = sc.playerid 
            AND o.gamesummaryid = sc.gamesummaryid
        WHERE g.season = {season}
        ORDER BY g.season, g.week;
        """
        
        player_data = execute_query(q)
        if player_data is not None and not player_data.empty:
            all_player_data.append(player_data)
    
    if not all_player_data:
        print("[WARNING] No player data found")
        return []
        
    player_df = pd.concat(all_player_data, ignore_index=True)
    
    # Add defensive opponent team id
    player_df["def_teamid"] = np.where(
        player_df["teamid"] == player_df["hometeamid"], 
        player_df["awayteamid"], 
        player_df["hometeamid"]
    )
    
    # Get top players by position for each team/game
    top_players = self._get_top_players_by_game(player_df)
    
    if top_players.empty:
        print("[WARNING] No top players identified")
        return []
    
    # Rolling window sizes (matching team features)
    roll_sizes = getattr(player_feature_configs_v2, 'rolling_windows', [5, 10])
    
    # Helper functions with snap count and bye week awareness
    def prior_expanding_snap_aware(series: pd.Series, season_series: pd.Series,
                                  week_series: pd.Series, team_series: pd.Series,
                                  snap_series: pd.Series, min_snaps: int = 5) -> pd.Series:
        """Calculate expanding mean with snap count and bye week validation"""
        s = series.shift(1 if exclude_current else 0)
        result = pd.Series(index=s.index, dtype=float)
        
        for team in team_series.unique():
            team_mask = team_series == team
            team_indices = s[team_mask].index
            
            for i, idx in enumerate(team_indices):
                prior_indices = team_indices[:i]
                if len(prior_indices) == 0:
                    result.loc[idx] = 0.0
                    continue
                
                valid_values = []
                for prior_idx in prior_indices:
                    prior_season = season_series.loc[prior_idx]
                    prior_week = week_series.loc[prior_idx]
                    prior_snaps = snap_series.loc[prior_idx] or 0
                    prior_value = s.loc[prior_idx]
                    
                    # Skip bye weeks
                    if self._is_bye_week(prior_season, prior_week, team, bye_schedule):
                        continue
                    
                    # Skip games with insufficient snaps (injury/rest)
                    if prior_snaps < min_snaps:
                        continue
                    
                    # Include all values, even zeros (bad games but player was active)
                    valid_values.append(prior_value)
                
                result.loc[idx] = np.mean(valid_values) if valid_values else 0.0
        
        return result
    
    def prior_rolling_snap_aware(series: pd.Series, window: int, season_series: pd.Series,
                                week_series: pd.Series, team_series: pd.Series, 
                                snap_series: pd.Series, min_snaps: int = 5) -> pd.Series:
        """Calculate rolling mean with snap count and bye week validation"""
        s = series.shift(1 if exclude_current else 0)
        result = pd.Series(index=s.index, dtype=float)
        
        for team in team_series.unique():
            team_mask = team_series == team
            team_indices = s[team_mask].index
            
            for i, idx in enumerate(team_indices):
                start_idx = max(0, i - window)
                prior_indices = team_indices[start_idx:i]
                
                if len(prior_indices) == 0:
                    result.loc[idx] = 0.0
                    continue
                
                valid_values = []
                for prior_idx in prior_indices:
                    prior_season = season_series.loc[prior_idx]
                    prior_week = week_series.loc[prior_idx]
                    prior_snaps = snap_series.loc[prior_idx] or 0
                    prior_value = s.loc[prior_idx]
                    
                    if self._is_bye_week(prior_season, prior_week, team, bye_schedule):
                        continue
                    if prior_snaps < min_snaps:
                        continue
                    
                    valid_values.append(prior_value)
                
                result.loc[idx] = np.mean(valid_values) if valid_values else 0.0
        
        return result
    
    # Create a complete game grid matching _hist_df structure
    game_grid = self._hist_df[['season', 'week', 'gamesummaryid', 'teamid', 'def_teamid']].copy()
    
    # Build features for each position rank
    all_player_features = {}
    built_player_features = []
    
    for player_rank in player_feature_configs_v2.position_rankings:
        rank_data = top_players[top_players['player_rank'] == player_rank].copy()
        
        if rank_data.empty:
            continue
            
        # Get ONLY relevant stats for this position
        if player_rank == 'QB1':
            relevant_stats = player_feature_configs_v2.get_position_stats('QB')
        elif player_rank == 'RB1':
            relevant_stats = player_feature_configs_v2.get_position_stats('RB')
        elif player_rank in ['WR1', 'WR2', 'WR3']:
            relevant_stats = player_feature_configs_v2.get_position_stats('WR')
        else:
            continue  # Skip unknown positions
        
        # Filter to only include stats that exist in the data AND are relevant
        stats_to_process = [stat for stat in relevant_stats if stat in rank_data.columns]
        
        if not stats_to_process:
            print(f"[WARNING] No relevant stats found for {player_rank}")
            continue
        
        # Merge with complete game grid to ensure all team/game combinations exist
        merge_columns = ['season', 'week', 'gamesummaryid', 'teamid']
        rank_complete = game_grid.merge(
            rank_data[merge_columns + stats_to_process + ['offensive_snaps']], 
            on=merge_columns, 
            how='left'
        )
        
        # Fill missing stats with 0 (player didn't play or wasn't top at position)
        for stat in stats_to_process:
            rank_complete[stat] = rank_complete[stat].fillna(0)
        rank_complete['offensive_snaps'] = rank_complete['offensive_snaps'].fillna(0)
        
        # Sort for proper time series (matching team features)
        rank_complete = rank_complete.sort_values(['season', 'week', 'gamesummaryid']).reset_index(drop=True)
        
        # Get position-specific snap threshold
        min_snaps = self._get_min_snaps_threshold(player_rank)
        
        # Extract series for calculations
        season_series = rank_complete['season']
        week_series = rank_complete['week']
        team_series = rank_complete['teamid']
        def_team_series = rank_complete['def_teamid']
        snap_series = rank_complete['offensive_snaps']
        
        # Create features for ONLY relevant stats
        for stat in stats_to_process:
            # Offensive features with snap awareness
            off_hist = prior_expanding_snap_aware(
                rank_complete[stat], season_series, week_series, 
                team_series, snap_series, min_snaps
            )
            feature_name = f"{player_rank.lower()}_{stat}__off_hist"
            all_player_features[feature_name] = off_hist
            built_player_features.append(feature_name)
            
            # Rolling windows with snap awareness
            for w in roll_sizes:
                off_roll = prior_rolling_snap_aware(
                    rank_complete[stat], w, season_series, week_series,
                    team_series, snap_series, min_snaps
                )
                feature_name = f"{player_rank.lower()}_{stat}__off_roll{w}"
                all_player_features[feature_name] = off_roll
                built_player_features.append(feature_name)
            
            # Defensive features if requested
            if include_defense:
                def_hist = prior_expanding_snap_aware(
                    rank_complete[stat], season_series, week_series,
                    def_team_series, snap_series, min_snaps
                )
                feature_name = f"{player_rank.lower()}_{stat}__def_hist"
                all_player_features[feature_name] = def_hist
                built_player_features.append(feature_name)
                
                # Defensive rolling windows
                for w in roll_sizes:
                    def_roll = prior_rolling_snap_aware(
                        rank_complete[stat], w, season_series, week_series,
                        def_team_series, snap_series, min_snaps
                    )
                    feature_name = f"{player_rank.lower()}_{stat}__def_roll{w}"
                    all_player_features[feature_name] = def_roll
                    built_player_features.append(feature_name)
                
                # Differentials if requested
                if include_differentials:
                    diff_hist = off_hist - def_hist
                    feature_name = f"{player_rank.lower()}_{stat}__diff_hist"
                    all_player_features[feature_name] = diff_hist
                    built_player_features.append(feature_name)
                    
                    for w in roll_sizes:
                        off_roll = prior_rolling_snap_aware(
                            rank_complete[stat], w, season_series, week_series,
                            team_series, snap_series, min_snaps
                        )
                        def_roll = prior_rolling_snap_aware(
                            rank_complete[stat], w, season_series, week_series,
                            def_team_series, snap_series, min_snaps
                        )
                        diff_roll = off_roll - def_roll
                        feature_name = f"{player_rank.lower()}_{stat}__diff_roll{w}"
                        all_player_features[feature_name] = diff_roll
                        built_player_features.append(feature_name)
    
    # Add all player features to _hist_df
    if all_player_features:
        for feature_name, feature_series in all_player_features.items():
            self._hist_df[feature_name] = feature_series.values
        
        # Update feature tracking (extend existing list)
        self._hist_features.extend(built_player_features)
        
        print(f"[INFO] Added {len(built_player_features)} player features with snap awareness")
        return built_player_features
    
    return []

def _load_bye_week_schedule(self, seasons):
    """Load bye week schedule for specified seasons"""
    seasons_str = ','.join(map(str, seasons))
    bye_query = f"""
    SELECT teamid, season, bye_week 
    FROM stats.byeweek 
    WHERE season IN ({seasons_str})
    """
    bye_df = execute_query(bye_query)
    
    if bye_df is None or bye_df.empty:
        return {}
    
    bye_schedule = {}
    for _, row in bye_df.iterrows():
        team, season, bye_week = row['teamid'], row['season'], row['bye_week']
        if team not in bye_schedule:
            bye_schedule[team] = {}
        bye_schedule[team][season] = bye_week
    
    return bye_schedule

def _is_bye_week(self, season, week, team_id, bye_schedule):
    """Check if specific team/season/week is a bye week"""
    return bye_schedule.get(team_id, {}).get(season) == week

def _get_min_snaps_threshold(self, player_rank):
    """Get minimum snaps threshold by position for injury/rest detection"""
    thresholds = {
        'QB1': 5,   # QBs should have meaningful participation
        'RB1': 3,   # RBs might have limited touches in some games  
        'WR1': 5,   # WRs need decent snap count to be relevant
        'WR2': 3,   # Secondary receivers might have fewer snaps
        'WR3': 3    # Third receivers often situational
    }
    return thresholds.get(player_rank, 5)

def _get_top_players_by_game(self, df: pd.DataFrame) -> pd.DataFrame:
    """Identify top players by position for each team/game."""
    
    position_rankings = []
    
    # Group by game and team
    for (season, week, gamesummaryid, teamid), team_game in df.groupby(['season', 'week', 'gamesummaryid', 'teamid']):
        
        # QB1 - top QB by pass attempts
        qbs = team_game[team_game['position'] == 'QB']
        if not qbs.empty:
            qb1 = qbs.nlargest(1, 'pass_att').copy()
            qb1['player_rank'] = 'QB1'
            position_rankings.append(qb1)
        
        # RB1 - top RB by rush attempts
        rbs = team_game[team_game['position'] == 'RB']
        if not rbs.empty:
            rb1 = rbs.nlargest(1, 'rush_att').copy()
            rb1['player_rank'] = 'RB1'
            position_rankings.append(rb1)
        
        # TE1 - top TE by targets
        tes = team_game[team_game['position'] == 'TE']
        if not tes.empty:
            te1 = tes.nlargest(1, 'targets').copy()
            te1['player_rank'] = 'TE1'
            position_rankings.append(te1)
        
        # WR1, WR2, WR3 - top 3 WRs by targets
        wrs = team_game[team_game['position'] == 'WR']
        if not wrs.empty:
            top_wrs = wrs.nlargest(3, 'targets').copy()
            for i, (idx, row) in enumerate(top_wrs.iterrows()):
                top_wrs.loc[idx, 'player_rank'] = f'WR{i+1}'
            position_rankings.append(top_wrs)
    
    if position_rankings:
        return pd.concat(position_rankings, ignore_index=True)
    else:
        return pd.DataFrame()
    


   