import pandas as pd
import numpy as np
from src.utils.db_utils import get_connection, execute_query
from src import config

class TeamLevelFeatures:
    def __init__(self):

        self._load_data()
        self._initialize_base_structures()
        self.parse_all_data()
        self.create_all_efficiency_stats()        
        self.team_stats_dict={}
        self.build_team_stats()
        
        # self.net_pass_yards_stat()
        # self.turnovers_stat()
        # self.total_yards_stat()
        # self.passing_efficency("completion efficiency")
        # self.passing_efficency("passing efficiency")
        # self.passing_efficency("receiving efficiency")
        # self.rushing_efficency("rushing efficiency")
        # self.rushing_efficency("rushing yards")
        # self.rushing_efficency("rushing tds")
        
        
    def _load_data(self):
        query = f"""
            SELECT *
            FROM stats.teamstats
            WHERE season > {config.model_config.start_season}
            """
        self.data=execute_query(query)
        self.data=self.data.sort_values(['season', 'week'])
    def _initialize_base_structures(self):
        self.teams=list(np.unique(self.data['teamid']))
        self.seasons=list(np.unique(self.data['season']))
        self.build_master_index()
        self.build_defense_id()
        self.stat_template=pd.DataFrame(index=self.master_index, columns=self.teams)
    def build_defense_id(self):
        home_team_id_idx=self.data.index[np.where(self.data['teamid']==self.data['hometeamid'])[0]]
        
        self.data['defenseid']=self.data['awayteamid']
        
        self.data.loc[home_team_id_idx, 'defenseid']=self.data.loc[home_team_id_idx, 'hometeamid']
    def build_master_index(self):
        df=self.data[['season', 'week']].astype(str)
        df=df.drop_duplicates(subset=['season', 'week'], keep='first')
        df['master index']=df['season']+'_'+df['week']
        self.master_index=df['master index'].values
    def _process_stat(self, current_stat):
        df_offense=self.stat_template
        df_defense=self.stat_template
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
    def parse_data(self, data_column, column_names):
        split_data = self.data[data_column].str.split('-', expand=True)
        
        # if data_column=="rush_yds_tds":
        #     split_data=split_data.iloc[:,:-1]
        
        split_data.columns=column_names
        
        self.data[column_names]=split_data
    
    def create_efficency_stats(self, numerator_column, denominator_column, stat):
        self.data[f'{stat}'] = np.where(
            self.data[denominator_column].astype(int) > 0, 
            self.data[numerator_column].astype(int) / self.data[denominator_column].astype(int), 
            np.nan
        )
    def parse_all_data(self):
        for i_raw, i_process in config.team_feature_configs.parse_data_colums.items():
            self.parse_data(i_raw, i_process)
    def create_all_efficiency_stats(self):
        for i_stat, i_inputs in config.team_feature_configs.efficiency_stats.items():
            self.create_efficency_stats(i_inputs[0], i_inputs[1], i_stat)
    def build_team_stats(self):
        for i_stat in config.team_feature_configs.team_features:
            self._process_stat(i_stat)
    def net_pass_yards_stat(self):
        
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_defense=pd.DataFrame(index=self.master_index, columns=self.teams)
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
        
    def turnovers_stat(self):
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_defense=pd.DataFrame(index=self.master_index, columns=self.teams)
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
    def total_yards_stat(self):
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_defense=pd.DataFrame(index=self.master_index, columns=self.teams)
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
    def passing_efficency(self, stat):
        def parse_conversion_passing(conv_str):

            completions, attempts, yards, tds, ints = map(int, conv_str.split('-'))
            return completions, attempts, yards, tds, ints
        data=self.data.loc[:, ['cmp_att_yd_td_int', 'week', 'teamid', 'defenseid']]
        self.data[['completions','attempts', 'yards', 'tds', 'int' ]]=data['cmp_att_yd_td_int'].apply(
        lambda x: pd.Series(parse_conversion_passing(x))
        )
        
        if stat=="completion efficiency":
            self.data[f'{stat}'] = np.where(
            self.data[f'attempts'] > 0, 
            self.data['completions'] / self.data['attempts'], 
            np.nan
        )
        if stat=="passing efficiency":
            self.data[f'{stat}'] = np.where(
                self.data[f'attempts'] > 0, 
                self.data['yards'] / self.data['attempts'], 
                np.nan
            )
        if stat=="receiving efficiency":
            self.data[f'{stat}'] = np.where(
                self.data[f'completions'] > 0, 
                self.data['yards'] / self.data['completions'], 
                np.nan
            )
        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_defense=pd.DataFrame(index=self.master_index, columns=self.teams)
        current_stat=stat
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
    def rushing_efficency(self,stat):

        def parse_conversion_passing(conv_str):
            try:
                rush_attempts, rush_yards, rush_tds = map(int, conv_str.split('-'))
            except: 
                print(conv_str)
                rush_attempts, rush_yards, rush_tds = 0,0,0
            return rush_attempts, rush_yards, rush_tds
        data=self.data.loc[:, ['rush_yds_tds', 'week', 'teamid', 'defenseid']]
        self.data[['rush_attempts','rushing yards', 'rushing tds']]=data['rush_yds_tds'].apply(
        lambda x: pd.Series(parse_conversion_passing(x))
        )
        
        if stat=="rushing efficiency":
            self.data[f'{stat}'] = np.where(
            self.data[f'rush_attempts'] > 0, 
            self.data['rushing yards'] / self.data['rush_attempts'], 
            np.nan
        )

        df_offense=pd.DataFrame(index=self.master_index, columns=self.teams)
        df_defense=pd.DataFrame(index=self.master_index, columns=self.teams)
        current_stat=stat
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
                
                df_defense.loc[master_index_loc.values, i_team]=mean_stat.values
        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive  {current_stat}']=df_offense.ffill().shift(1)                    
        
