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
        self.create_spread_dict()
        
        
        
    def create_spread_dict(self):
        
        week_list=list(pd.DataFrame(self.game_index).drop_duplicates().values.flatten())
        df=self.game_data
        df.index=self.game_index
        self.spread_dict={}
        for i_week in week_list:

            if type(df.loc[i_week,'spread']) != np.int64:
                spread_list=list(df.loc[i_week,'spread'])
                home_team_list=list(df.loc[i_week,'hometeamid'])
                away_team_list=list(df.loc[i_week,'awayteamid'])
      
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
            for i_game_week in self.spread_dict.keys():
                if len(self.spread_dict[i_game_week][0][0])>1:
                    for i_game in range(len(self.spread_dict[i_game_week][0])):
                    
                        df.loc[f"{i_game_week}_{i_game}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][0][i_game]]
                        df.loc[f"{i_game_week}_{i_game}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][1][i_game]]
                        df.loc[f"{i_game_week}_{i_game}", "spread"]=self.spread_dict[i_game_week][2][i_game]
                else:
                    print(i_game_week)
                    df.loc[f"{i_game_week}_{i_game}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][0]]
                    df.loc[f"{i_game_week}_{i_game}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][1]]
                    df.loc[f"{i_game_week}_{i_game}", "spread"]=self.spread_dict[i_game_week][2]                       

        self.x_y=df
            
        

        
        
        

