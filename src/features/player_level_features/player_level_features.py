"""Player level features implementation for NFLModelV2 integration.

Add this method to your NFLModelV2 class to integrate player features.
"""

def build_player_feature_matrices(self, exclude_current: bool = True, include_defense: bool = True, include_differentials: bool = True):
    """
    Build player-level rolling average features and add to existing team features.
    
    Creates features for top players by position:
    - QB1 (top QB by pass attempts)
    - RB1 (top RB by rush attempts) 
    - TE1 (top TE by targets)
    - WR1, WR2, WR3 (top 3 WRs by targets)
    
    Args:
        exclude_current: If True, exclude current game from rolling averages
        include_defense: If True, include defensive (opponent) player features
        include_differentials: If True, include offensive - defensive differentials
        
    Returns:
        List of player feature column names created
    """
    if not hasattr(self, "_hist_df"):
        raise RuntimeError("Call build_feature_matrices first to create team features.")
    
    from src.config.config_v2 import player_feature_configs_v2
    from src.utils.db_utils import execute_query
    
    # Get unique seasons from existing games
    seasons = self._hist_df['season'].unique()
    
    # Load player data for these seasons
    all_player_data = []
    for season in seasons:
        q = f"""
        SELECT 
            g.season, 
            g.week, 
            o.playerid, 
            o.player, 
            p.current_position as position,
            o.teamid, 
            o.gamesummaryid,
            g.hometeamid,
            g.awayteamid,
            -- Passing stats
            COALESCE(o.pass_yds, 0) as pass_yds,
            COALESCE(o.pass_td, 0) as pass_td,
            COALESCE(o.pass_int, 0) as pass_int,
            COALESCE(o.pass_cmp, 0) as pass_cmp,
            COALESCE(o.pass_att, 0) as pass_att,
            o.pass_rating,
            COALESCE(o.pass_long, 0) as pass_long,
            COALESCE(o.pass_sacked, 0) as pass_sacked,
            COALESCE(o.pass_sacked_yds, 0) as pass_sacked_yds,
            -- Rushing stats  
            COALESCE(o.rush_yds, 0) as rush_yds,
            COALESCE(o.rush_td, 0) as rush_td,
            COALESCE(o.rush_att, 0) as rush_att,
            COALESCE(o.rush_long, 0) as rush_long,
            -- Receiving stats
            COALESCE(o.rec_yds, 0) as rec_yds,
            COALESCE(o.rec_td, 0) as rec_td,
            COALESCE(o.rec, 0) as rec,
            COALESCE(o.targets, 0) as targets,
            COALESCE(o.rec_long, 0) as rec_long,
            -- Other
            COALESCE(o.fumbles, 0) as fumbles,
            COALESCE(o.fumbles_lost, 0) as fumbles_lost
        FROM stats.offense o
        JOIN stats.gamesummary g ON o.gamesummaryid = g.gamesummaryid
        JOIN stats.player p ON o.playerid = p.playerid
        WHERE g.season = {season}
        ORDER BY g.season, g.week, o.gamesummaryid, o.teamid;
        """
        
        player_data = execute_query(q)
        if player_data is not None and not player_data.empty:
            all_player_data.append(player_data)
    
    if not all_player_data:
        print("[WARNING] No player data found for seasons")
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
    
    # Rolling window sizes and helper functions
    roll_sizes = player_feature_configs_v2.rolling_windows
    
    def prior_expanding(series: pd.Series) -> pd.Series:
        s = series.shift(1 if exclude_current else 0)
        return s.expanding(min_periods=1).mean()
    
    def prior_rolling(series: pd.Series, w: int) -> pd.Series:
        s = series.shift(1 if exclude_current else 0)
        return s.rolling(window=w, min_periods=1).mean()
    
    # Build features for each position rank
    player_feature_frames = []
    built_player_features = []
    
    for player_rank in player_feature_configs_v2.position_rankings:
        rank_data = top_players[top_players['player_rank'] == player_rank].copy()
        
        if rank_data.empty:
            continue
            
        # Get relevant stats for this position
        if player_rank == 'QB1':
            stats = player_feature_configs_v2.get_position_stats('QB')
        elif player_rank == 'RB1':
            stats = player_feature_configs_v2.get_position_stats('RB')
        elif player_rank == 'TE1':
            stats = player_feature_configs_v2.get_position_stats('TE')
        else:  # WR1, WR2, WR3
            stats = player_feature_configs_v2.get_position_stats('WR')
        
        # Sort by season, week for proper time series
        rank_data = rank_data.sort_values(['season', 'week']).reset_index(drop=True)
        
        # Create a complete game grid to ensure all team/game combinations exist
        game_grid = self._hist_df[['season', 'week', 'gamesummaryid', 'teamid']].drop_duplicates()
        rank_data = game_grid.merge(
            rank_data, 
            on=['season', 'week', 'gamesummaryid', 'teamid'], 
            how='left'
        )
        
        # Fill missing stats with 0 (player didn't play or wasn't top at position)
        for stat in stats:
            if stat in rank_data.columns:
                rank_data[stat] = rank_data[stat].fillna(0)
        
        # Group by team for rolling calculations
        team_grp = rank_data.groupby('teamid', group_keys=False)
        def_grp = rank_data.groupby('def_teamid', group_keys=False) if include_defense else None
        
        # Create features for each stat
        for stat in stats:
            if stat not in rank_data.columns:
                continue
                
            stat_data = pd.to_numeric(rank_data[stat], errors='coerce').fillna(0)
            
            # Offensive features
            off_hist = team_grp[stat].apply(prior_expanding)
            cols = {f"{player_rank.lower()}_{stat}__off_hist": off_hist}
            
            # Rolling windows
            for w in roll_sizes:
                off_roll = team_grp[stat].apply(lambda s: prior_rolling(s, w))
                cols[f"{player_rank.lower()}_{stat}__off_roll{w}"] = off_roll
            
            # Defensive features if requested
            if include_defense and def_grp is not None:
                def_hist = def_grp[stat].apply(prior_expanding)
                cols[f"{player_rank.lower()}_{stat}__def_hist"] = def_hist
                
                # Defensive rolling windows
                for w in roll_sizes:
                    def_roll = def_grp[stat].apply(lambda s: prior_rolling(s, w))
                    cols[f"{player_rank.lower()}_{stat}__def_roll{w}"] = def_roll
                
                # Differentials if requested
                if include_differentials:
                    cols[f"{player_rank.lower()}_{stat}__diff_hist"] = off_hist - def_hist
                    for w in roll_sizes:
                        off_roll = team_grp[stat].apply(lambda s: prior_rolling(s, w))
                        def_roll = def_grp[stat].apply(lambda s: prior_rolling(s, w))
                        cols[f"{player_rank.lower()}_{stat}__diff_roll{w}"] = off_roll - def_roll
            
            # Add to feature list
            built_player_features.extend(cols.keys())
            
            # Create DataFrame for this stat's features
            stat_features = pd.DataFrame(cols, index=rank_data.index)
            player_feature_frames.append(stat_features)
    
    # Combine all player features
    if player_feature_frames:
        player_features_df = pd.concat(player_feature_frames, axis=1)
        
        # Add player features to existing hist_df
        # Match on the game identifier columns
        merge_cols = ['season', 'week', 'gamesummaryid', 'teamid']
        
        # Create merge key for existing data
        existing_key = self._hist_df[merge_cols].reset_index(drop=True)
        player_key = rank_data[merge_cols].reset_index(drop=True)
        
        # Merge player features
        combined_features = existing_key.merge(
            pd.concat([player_key, player_features_df], axis=1),
            on=merge_cols,
            how='left'
        )
        
        # Fill any missing player features with 0
        feature_cols = [c for c in combined_features.columns if c not in merge_cols]
        for col in feature_cols:
            if col in combined_features.columns:
                combined_features[col] = combined_features[col].fillna(0)
        
        # Add player features to existing hist_df
        for col in feature_cols:
            if col in combined_features.columns:
                self._hist_df[col] = combined_features[col].values
        
        # Update feature tracking
        if hasattr(self, '_hist_features'):
            self._hist_features.extend(built_player_features)
        else:
            self._hist_features = built_player_features.copy()
        
        print(f"[INFO] Added {len(built_player_features)} player features")
        return built_player_features
    
    return []

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
    


    ##Test