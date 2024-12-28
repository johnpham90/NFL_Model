import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query
from src import config

class TeamLevelFeatures:
    def __init__(self):
        query = f"""
            SELECT *
            FROM stats.teamstats
            WHERE season > {config.model_config.start_season}
            """
        self.data=execute_query(query)
        self.data=self.data.sort_values(['season', 'week'])
        self.teams=list(np.unique(self.data['teamid']))
        self.seasons=list(np.unique(self.data['season']))
        self.build_defense_id()
        self.build_master_index()
        
        self.team_stats_dict={}
        
        self.net_pass_yards_stat()
        self.turnovers_stat()
        self.total_yards_stat()
        
        
    def build_defense_id(self):
        home_team_id_idx=self.data.index[np.where(self.data['teamid']!=self.data['hometeamid'])[0]]
        
        self.data['defenseid']=self.data['awayteamid']
        
        self.data.loc[home_team_id_idx, 'defenseid']=self.data.loc[home_team_id_idx, 'hometeamid']
    def build_master_index(self):
        df=self.data[['season', 'week']].astype(str)
        df=df.drop_duplicates(subset=['season', 'week'], keep='first')
        df['master index']=df['season']+'_'+df['week']
        self.master_index=df['master index'].values

    def net_pass_yards_stat(self):
        
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_deffense=pd.DataFrame(index=self.master_index, columns=self.teams)
        current_stat='net_pass_yards'
        
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['teamid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_offense.loc[master_index_loc.values, i_team]=mean_stat.values
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['defenseid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_deffense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'deffensive {current_stat}']=df_deffense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
        
    def turnovers_stat(self):
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_deffense=pd.DataFrame(index=self.master_index, columns=self.teams)
        current_stat='turnovers'
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['teamid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_offense.loc[master_index_loc.values, i_team]=mean_stat.values
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['defenseid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_deffense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'deffensive {current_stat}']=df_deffense.ffill().shift(1)
        self.team_stats_dict[f'offensive net_pass_yards {current_stat}']=df_offense.ffill().shift(1)
    def total_yards_stat(self):
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_deffense=pd.DataFrame(index=self.master_index, columns=self.teams)
        current_stat='total_yards'
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['teamid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_offense.loc[master_index_loc.values, i_team]=mean_stat.values
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['defenseid']==i_team) & (self.data['season']==i_season))[0]
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                current_data=self.data.iloc[current_season_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                
                df_deffense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'deffensive {current_stat}']=df_deffense.ffill().shift(1)
        self.team_stats_dict[f'offensive net_pass_yards {current_stat}']=df_offense.ffill().shift(1)
        
        
def offensive_mettrics(season, stat):
    
    query=f"""SELECT *
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results=execute_query(query)

    weeks=np.unique(query_results.loc[:,"week"].values)
    teams=np.unique(query_results.loc[:,"teamid"].values)

    stats_pg_df=pd.DataFrame(index=weeks, columns=teams)

    for i_week in weeks:
        current_week_idx=np.where(query_results["week"]==i_week)[0]
        current_week_df=pd.DataFrame(query_results.iloc[current_week_idx]).set_index('teamid')
        stats_pg_df.loc[i_week, current_week_df.index]=current_week_df.loc[:, stat]
        
    average_stats_pg_df=pd.DataFrame(index=weeks, columns=teams)
    
    for i_week in range(weeks.shape[0]):
        for i_team in teams:
            current_week=weeks[i_week]
            average_stats_pg_df.loc[current_week,i_team]=np.mean(stats_pg_df.loc[:current_week, i_team].dropna())
        
    return stats_pg_df, average_stats_pg_df


def defensive_metrics(season, stat):
    """
    Calculate yards allowed per game (YPG) and the rolling average yards allowed throughout the season
    based on the specified stat, from the defensive team's perspective.

    Args:
    - season (int): The season to query.
    - stat (str): The stat to calculate (e.g., 'total_yards', 'passing_yards').

    Returns:
    - pd.DataFrame: Yards allowed for each game by week and team.
    - pd.DataFrame: Rolling average yards allowed per game for each team.
    """
    query = f"""
    SELECT *
    FROM stats.teamstats
    WHERE season = {season}
    """
    query_results = execute_query(query)

    # Ensure data is sorted by team and week for rolling averages
    query_results = query_results.sort_values(by=["teamid", "week"])

    # Add a column for the defensive team's metrics
    def assign_defensive_stats(row):
        if row["teamid"] == row["hometeamid"]:
            # If the stat belongs to the home team, the defense is the away team
            defensive_team = row["awayteamid"]
        elif row["teamid"] == row["awayteamid"]:
            # If the stat belongs to the away team, the defense is the home team
            defensive_team = row["hometeamid"]
        else:
            raise ValueError("Invalid teamid comparison")
        
        # Return the stat as the defensive metric for the opposing team
        return defensive_team, row[stat]

    # Apply the logic to calculate defensive metrics
    query_results[["defensive_team", "stats_allowed"]] = query_results.apply(
        lambda row: assign_defensive_stats(row), axis=1, result_type="expand"
    )

    # Aggregate stats by defensive team
    defensive_stats_df = query_results.pivot(index="week", columns="defensive_team", values="stats_allowed")

    # Calculate rolling average of stats allowed
    query_results["stats_allowed_avg"] = (
        query_results.groupby("defensive_team")["stats_allowed"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    average_pg_df = query_results.pivot(index="week", columns="defensive_team", values="stats_allowed_avg")

    return defensive_stats_df, average_pg_df
