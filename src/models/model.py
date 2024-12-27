from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.utils.db_utils import get_connection, execute_query
import pandas as pd
import numpy as np
import pickle

from src.features import team_level_features_class

class NFLPredictor:
    def __init__(self):
        
        query = f"""
        SELECT *
        FROM stats.gamesummary
        """
        self.game_data = execute_query(query)
        self.game_data=self.game_data.sort_values(['season', 'week'])
        self.game_data['spread']=self.game_data['homescore'].values-self.game_data['awayscore'].values
        self.team_feautres=team_level_features_class.TeamLevelFeatures()
        self.team_x_variables=self.team_feautres.team_stats_dict
        
        self.game_index=(self.game_data['season'].astype(str)+'_'+self.game_data['week'].astype(str)).values
        
    def create_spread_dict(self):
        
        week_list=list(pd.DataFrame(self.game_index).drop_duplicates().values.flatten())
        df=self.game_data
        df.index=self.game_index
        self.spread_dict={}
        for i_week in week_list:
            print(i_week)
            if type(df.loc[i_week,'spread']) != np.int64:
                spread_list=list(df.loc[i_week,'spread'])
                home_team_list=list(df.loc[i_week,'hometeam'])
                away_team_list=list(df.loc[i_week,'awayteam'])
      
                self.spread_dict[i_week] = [home_team_list, away_team_list, spread_list]
            else:
                self.spread_dict[i_week]=[home_team_list[0], away_team_list[0], spread_list[0]]
    def build_x_y_variable(self):
        feature_list=[]
        for i_feature in self.team_feautres.team_stats_dict.keys():
            feature_list.append(f'home team {i_feature}')
            feature_list.append(f'away team {i_feature}')
        feature_list.append('spread')
        df=pd.DataFrame(columns=feature_list) 
        game_data=pd.DataFrame(self.game_data)
        game_data.index=self.game_index
        
        for i_feature in self.team_feautres.team_stats_dict.keys():
            current_feature_data=self.team_feautres.team_stats_dict[i_feature]
            for i_game_week in self.team_feautres.master_index:
                data=self.game_data.loc[i_game_week]
                home_team=data['hometeamid']
                away_team=data['awayteamid']
                spread=data['spread']
                game_cntr=0
                for i_game in range(data.shape[0]):
                    if i_game_week=="2004_21":
                        a=1
                    print(i_game_week,i_game,game_cntr )
                    if type(home_team)!=str:
                        df.loc[f"{i_game_week}_{game_cntr}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, home_team[i_game]]
                        df.loc[f"{i_game_week}_{game_cntr}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, away_team[i_game]]
                        df.loc[f"{i_game_week}_{game_cntr}", "spread"]=spread[i_game]
                    else:
                        df.loc[f"{i_game_week}_{game_cntr}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, home_team]
                        df.loc[f"{i_game_week}_{game_cntr}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, away_team]
                        df.loc[f"{i_game_week}_{game_cntr}", "spread"]=spread
                    game_cntr+=1
        self.x_y=df
            
        
    # def build_spread_variable(self):
        
    # def build_data_set(self):
    
    #     feature_names=[]

    #     for i_feature in self.team_x_variables.keys():
    #         feature_names.append(f"home team {i_feature}")
    #         feature_names.append(f"away team {i_feature}")
    #     feature_names.append("spread")   
    #     x_features_df=pd.DataFrame(columns=feature_names)
    #     all_stat_dict={}
    #     for i_stat in stats:
    #         stat_list=[]
    #         for i_season in training_seasons:

    #             yards_pg, average_stats_pg=team_level_features.generic_team_stats_per_game(i_season, i_stat)
    #             stat_list.append(average_stats_pg)
    #             y_out=build_spread_variable(i_season)
                
    #             keys=y_out.keys()
    #             weeks=list(keys)
                
    #             for i_week in weeks:
    #                 for i_game in range(len(y_out[i_week][0])):
    #                     home_team=y_out[i_week][0][i_game]
    #                     away_team=y_out[i_week][1][i_game]
    #                     x_features_df.loc[f"{i_season}_{i_week}_{i_game}",f"home team {i_stat}"]=average_stats_pg.loc[i_week, home_team]
    #                     x_features_df.loc[f"{i_season}_{i_week}_{i_game}",f"away team {i_stat}"]=average_stats_pg.loc[i_week, away_team]
    #                     x_features_df.loc[f"{i_season}_{i_week}_{i_game}","spread"]=y_out[i_week][2][i_game]
                        
                        
                
    #         current_stat_df=pd.concat(stat_list, axis=0)
    #         all_stat_dict[i_stat]=[current_stat_df]
    # return x_features_df
        
        
        

