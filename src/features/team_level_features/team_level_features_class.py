import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.db_utils import get_connection, execute_query
from src.utils import config


class TeamLevelFeatures:
    def __init__(self):
        self.average_windows=[2,5,10]
        self._load_data()
        self._initialize_base_structures()
        self.parse_all_data()
        self.create_all_efficiency_stats()
        self.add_team_score()        
        self.team_stats_dict={}
        self.single_stats_dict={}
        self.build_team_stats()
        self.normalize_features()
        # self.normalize_strength_of_schedule()
        
        
    def _load_data(self):
        """this function loads the data that will be needed to generate teamlevel features
        """        
        query = f"""
            SELECT gs.gamesummaryid, gs.awayscore, gs.homescore, ts.*
            FROM stats.gamesummary gs
            join stats.teamstats ts on gs.gamesummaryid = ts.gamesummaryid
            WHERE ts.season > {config.model_config.start_season}
                and gs.season > {config.model_config.start_season}
            """
        self.data=execute_query(query)
        self.data=self.data.sort_values(['season', 'week'])
        
        bad_data=np.where(self.data["rush_yds_tds"]=="8--1-0")[0]
        rush_yds_tds_idx=self.data.columns.get_loc("rush_yds_tds")
        self.data.iloc[bad_data, rush_yds_tds_idx]="8-1-0"
        self.data = self.data.T.drop_duplicates().T
    def _initialize_base_structures(self):
        """a function to build the structures needed for processing all of the data
        """        
        self.teams=list(np.unique(self.data['teamid']))
        self.seasons=list(np.unique(self.data['season']))
        self.build_master_index()
        self.build_defense_id()
        self.stat_template=pd.DataFrame(index=self.master_index, columns=self.teams)
    def build_defense_id(self):
        """builds deffensive id to create deffensive stats
        """        
        home_team_id_idx=self.data.index[np.where(self.data['teamid']==self.data['hometeamid'])[0]]
        
        self.data['defenseid']=self.data['hometeamid']
        
        self.data.loc[home_team_id_idx, 'defenseid']=self.data.loc[home_team_id_idx, 'awayteamid']
    def build_master_index(self):
        """builds a master index to make locating data easier
        """        
        df=self.data[['season', 'week']].astype(str)
        df=df.drop_duplicates(subset=['season', 'week'], keep='first')
        df['master index']=df['season']+'_'+df['week']
        self.master_index=df['master index'].values
    def _process_stat(self, current_stat):
        """creates an offensive and deffensive stat for all teams and all games and adds it to team_stats_dict

        Args:
            current_stat (str): stat thats being processed
        """        
        df_offense=self.stat_template.copy()
        df_defense=self.stat_template.copy()
        df_single_game_stat=self.stat_template.copy()
        for i_team in self.teams:
            for i_season in self.seasons:
                current_season_idx=np.where((self.data['teamid']==i_team) & (self.data['season']==i_season))[0]
                current_season_def_idx=np.where((self.data['defenseid']==i_team) & (self.data['season']==i_season))[0]
                
                master_index_loc=self.data.loc[self.data.index[current_season_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_idx],'week'].astype(str)
                master_index_def_loc=self.data.loc[self.data.index[current_season_def_idx], 'season'].astype(str)+'_'+self.data.loc[self.data.index[current_season_def_idx],'week'].astype(str)
                
                current_data=self.data.iloc[current_season_idx]
                current_data_def=self.data.iloc[current_season_def_idx]
                
                mean_stat=current_data.loc[:, current_stat].expanding().mean()
                mean_stat_def=current_data_def.loc[:, current_stat].expanding().mean()
                
                
                df_offense.loc[master_index_loc.values, i_team]=mean_stat.values
                df_defense.loc[master_index_def_loc.values, i_team]=mean_stat_def.values
                df_single_game_stat.loc[master_index_loc.values, i_team]=current_data.loc[:, current_stat].values

        self.team_stats_dict[f'defensive {current_stat}']=df_defense.ffill().shift(1)
        self.team_stats_dict[f'offensive {current_stat}']=df_offense.ffill().shift(1)
        self.single_stats_dict[current_stat]=df_single_game_stat
    def build_defense_df(self):
        self.df_matchups=self.stat_template.copy()
        for i_team in self.teams:
            team_index=np.where(self.data['teamid']==i_team)[0]
            season_loc=self.data.columns.get_loc('season')
            week_loc=self.data.columns.get_loc('week')
            defenseid_loc=self.data.columns.get_loc('defenseid')
            team_master_index=self.data.iloc[team_index, season_loc].astype(str)+'_'+self.data.iloc[team_index, week_loc].astype(str)
            
            opposing_defense=self.data.iloc[team_index, defenseid_loc]
            
            self.df_matchups.loc[team_master_index, i_team]=opposing_defense.values
            
    def normalize_strength_of_schedule(self):
        self.build_defense_df()
        
        for i_stat in tqdm(config.team_feature_configs.team_features):
            single_game_data=self.single_stats_dict[i_stat]
            defense_data=self.team_stats_dict[f'defensive {i_stat}']
            normalized_feature=self.stat_template.copy()
            for i_game_week in self.df_matchups.index[1:]:
                for i_team in self.df_matchups.columns:
                    if defense_data.loc[i_game_week, i_team] !=0 and np.isfinite(float(defense_data.loc[i_game_week, i_team])) and np.isfinite(float(single_game_data.loc[i_game_week, i_team])):
                        normalized_feature.loc[i_game_week, i_team]=float(single_game_data.loc[i_game_week, i_team])/defense_data.loc[i_game_week, i_team]
                    else:
                        normalized_feature.loc[i_game_week, i_team]=0
            self.single_stats_dict[f'normalized {i_stat}']=normalized_feature
            
            for i_window in self.average_windows:
                self.team_stats_dict[f'normalize {i_stat} {i_window}']=normalized_feature.rolling(window=i_window).mean().shift(1)

    def parse_data(self, data_column, column_names):
        """parses data thats gotta multiple data points in one column

        Args:
            data_column (_type_): raw data column that needs to be parsed
            column_names (_type_): names of the columns of the parsed data
        """        
        split_data = self.data[data_column].str.split('-', expand=True)
        
             
        split_data.columns=column_names
        
        self.data[column_names]=split_data
    
    def create_efficency_stats(self, numerator_column, denominator_column, stat):
        """creates an efficency stat given two data columns

        Args:
            numerator_column (str): data point in the numerator
            denominator_column (str): data point in the denominator
            stat (str): name of the stat being calculated
        """        
        self.data[f'{stat}'] = np.where(
            self.data[denominator_column].astype(int) > 0, 
            self.data[numerator_column].astype(int) / self.data[denominator_column].astype(int), 
            np.nan
        )
    def parse_all_data(self):
        """parses all necessary data points from the config file
        """        
        for i_raw, i_process in config.team_feature_configs.parse_data_colums.items():
            self.parse_data(i_raw, i_process)
    def create_all_efficiency_stats(self):
        """creates all the efficiency stats from the config file
        """        
        print("creating efficiency stats")
        for i_stat, i_inputs in tqdm(config.team_feature_configs.efficiency_stats.items()):
            self.create_efficency_stats(i_inputs[0], i_inputs[1], i_stat)
    def build_team_stats(self):
        """creates all the team stats from the data file
        """
        print("building team stats")        
        for i_stat in tqdm(config.team_feature_configs.team_features):
            self._process_stat(i_stat)
    def add_team_score(self):
        home_team_idx=np.where(self.data["teamid"]==self.data["hometeamid"])[0]
        home_score_position = self.data.columns.get_loc("homescore")
        self.data["team points"]=0
        self.data.iloc[home_team_idx, -1]=self.data.iloc[home_team_idx,home_score_position]
        away_team_idx=np.where(self.data["teamid"]==self.data["awayteamid"])[0]
        away_score_position = self.data.columns.get_loc("awayscore")
        self.data.iloc[away_team_idx, -1]=self.data.iloc[away_team_idx,away_score_position]
    def normalize_features(self):
        features_to_normalize=list(self.team_stats_dict.keys()).copy()
        for i_feature in features_to_normalize:
            df=self.stat_template.copy()
            for i_team in df.columns:
                df[i_team]=(self.team_stats_dict[i_feature][i_team]-np.mean(self.team_stats_dict[i_feature], axis=1))/np.std(self.team_stats_dict[i_feature], axis=1)
            
            self.team_stats_dict[f"{i_feature}_normalized"]=df