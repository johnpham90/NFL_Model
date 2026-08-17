"""DEPRECATED: This module is deprecated. Use xgboost_randomforrest_model_v2.py instead.

Migration:
    from src.models.xgboost_randomforrest_model_v2 import NFLModelV2
    model = NFLModelV2(target="spread")
"""
import warnings
warnings.warn(
    "xgboost_randomforrest_model.py is deprecated. "
    "Use xgboost_randomforrest_model_v2.NFLModelV2 instead.",
    DeprecationWarning,
    stacklevel=2
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from src.utils.db_utils import get_connection, execute_query
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.features.team_level_features import team_level_features_class
from src.config import config

class NFLPredictor:
    def __init__(self):
        
        query = f"""
        SELECT *
        FROM stats.gamesummary
        WHERE season > {config.model_config.start_season}
        """
        self.game_data = execute_query(query)
        self.game_data=self.game_data.sort_values(['season', 'week'])
        self.game_data['spread']=self.game_data['homescore'].values-self.game_data['awayscore'].values
        self.team_feautres=team_level_features_class.TeamLevelFeatures()
        self.team_x_variables=self.team_feautres.team_stats_dict
        self.game_index=(self.game_data['season'].astype(str)+'_'+self.game_data['week'].astype(str)).values
        self.create_labels_dict()
        self.random_state=42
        
        
        

    def create_labels_dict(self):
        
        week_list=list(pd.DataFrame(self.game_index).drop_duplicates().values.flatten())

        self.game_data["binary_ou_label"]=np.where(self.game_data["overunderresults"]=="Over",1,0)
        self.game_data["binary_spread_label"]=np.where(self.game_data['spreadfavoriteteam']==self.game_data['spreadteamcovered'],1,0)
        self.game_data['total points']=self.game_data['homescore'].values+self.game_data['awayscore'].values
        df=self.game_data
        df.index=self.game_index
        self.spread_dict={}
        for i_week in week_list:

            if type(df.loc[i_week,'spread']) != np.int64:
                spread_list=list(df.loc[i_week,'spread'])
                binary_ou_list=list(df.loc[i_week,"binary_ou_label"])
                binary_spread_list=list(df.loc[i_week,"binary_spread_label"])
                total_points_list=list(df.loc[i_week,'total points'])
                home_team_list=list(df.loc[i_week,'hometeamid'])
                away_team_list=list(df.loc[i_week,'awayteamid'])
                game_summary_id_list=list(df.loc[i_week, "gamesummaryid"])
      
                self.spread_dict[i_week] = [home_team_list, away_team_list, spread_list, total_points_list, binary_spread_list, binary_ou_list, game_summary_id_list]
            else:
                spread_list=df.loc[i_week,'spread']
                binary_ou_list=df.loc[i_week,"binary_ou_label"]
                binary_spread_list=df.loc[i_week,"binary_spread_label"]
                total_points_list=df.loc[i_week,'total points']
                home_team_list=df.loc[i_week,'hometeamid']
                away_team_list=df.loc[i_week,'awayteamid']
                game_summary_id_list=df.loc[i_week, "gamesummaryid"]
                self.spread_dict[i_week]=[home_team_list, away_team_list, spread_list, total_points_list, binary_spread_list, binary_ou_list, game_summary_id_list]
    
    def build_x_y_variable(self):
        self.feature_list=[]
        self.labels_list=[]
        for i_feature in self.team_feautres.team_stats_dict.keys():
            self.feature_list.append(f'home team {i_feature}')
            self.feature_list.append(f'away team {i_feature}')
        self.labels_list.append('spread')
        self.labels_list.append('total points')
        self.labels_list.append("binary_spread_label")
        self.labels_list.append("binary_ou_label")
        self.feature_list.append("gamesummaryid")
        self.labels_list.append("gamesummaryid")
        game_data=pd.DataFrame(self.game_data)
        game_data.index=self.game_index
        
        
        df_x=pd.DataFrame(columns=self.feature_list)
        df_y=pd.DataFrame(columns=self.labels_list)
        
        print("building X Y Dataframes")
        for i_feature in tqdm(self.team_feautres.team_stats_dict.keys()):
            
            current_feature_data=self.team_feautres.team_stats_dict[i_feature]
            for i_game_week in self.spread_dict.keys():
                if len(self.spread_dict[i_game_week][0][0])>1:
                    for i_game in range(len(self.spread_dict[i_game_week][0])):
                        
                        

                        df_x.loc[f"{i_game_week}_{i_game}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][0][i_game]]
                        df_x.loc[f"{i_game_week}_{i_game}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][1][i_game]]
                        df_y.loc[f"{i_game_week}_{i_game}", "spread"]=self.spread_dict[i_game_week][2][i_game]
                        df_y.loc[f"{i_game_week}_{i_game}", "total points"]=self.spread_dict[i_game_week][3][i_game]
                        df_y.loc[f"{i_game_week}_{i_game}", "binary_spread_label"]=self.spread_dict[i_game_week][4][i_game]
                        df_y.loc[f"{i_game_week}_{i_game}", "binary_ou_label"]=self.spread_dict[i_game_week][5][i_game]
                        df_x.loc[f"{i_game_week}_{i_game}", "gamesummaryid"]=self.spread_dict[i_game_week][6][i_game]
                        df_y.loc[f"{i_game_week}_{i_game}", "gamesummaryid"]=self.spread_dict[i_game_week][6][i_game]
                else:

                    df_x.loc[f"{i_game_week}_0", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][0]]
                    df_x.loc[f"{i_game_week}_0", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][1]]
                    df_y.loc[f"{i_game_week}_0", "spread"]=self.spread_dict[i_game_week][2]   
                    df_y.loc[f"{i_game_week}_0", "total points"]=self.spread_dict[i_game_week][3]
                    df_y.loc[f"{i_game_week}_0", "binary_spread_label"]=self.spread_dict[i_game_week][4]
                    df_y.loc[f"{i_game_week}_0", "binary_ou_label"]=self.spread_dict[i_game_week][5]
                    df_y.loc[f"{i_game_week}_0", "gamesummaryid"]=self.spread_dict[i_game_week][6]
                    df_x.loc[f"{i_game_week}_0", "gamesummaryid"]=self.spread_dict[i_game_week][6]                    

        self.df_x=df_x
        self.df_y=df_y
        
    def build_single_game_features(self, home_team, away_team):
        df=pd.DataFrame(columns=self.feature_list) 
        for i_feature in self.team_feautres.team_stats_dict.keys():
            current_feature_data=self.team_feautres.team_stats_dict[i_feature].loc[:,[home_team, away_team]]

                    
            df.loc[f"{home_team}_{away_team}", f"home team {i_feature}"]=current_feature_data.iloc[-1, 0]
            df.loc[f"{home_team}_{away_team}", f"away team {i_feature}"]=current_feature_data.iloc[-1, 1]
            
        return df


    def build_xg_boost_model(self,x,y, param_distirbution, model_name, model):        
        x_train, x_test, y_train,y_test=train_test_split(x,y, test_size=config.model_config.train_test_split, shuffle=False)
        
        
        model_parameters=RandomizedSearchCV(model(random_state=42), param_distirbution,n_jobs=-1,cv=5).fit(x_train.to_numpy(), y_train.to_numpy()).best_params_
        model_parameters['random_state'] = 42  
        xg_model=model(**model_parameters).fit(X=x_train.to_numpy(), y=y_train.to_numpy())
        xg_model.save_model(f'{model_name}.json')
    
    def model_prediction(self, model,x=np.array([]),plot_feature_importance=False, plot_scatter=False):
        if x.shape[0]==0:
            self.predictions=model.predict(self.x_test)
        else:
            self.predictions=model.predict(x)
        
        if plot_feature_importance:
            feature_importance = pd.DataFrame({
            'feature': self.x_y.iloc[:,:-1].columns,
            'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)


        # Create the plot
            plt.figure(figsize=(10, 15))
            sns.barplot(data=feature_importance.iloc[:50], x='importance', y='feature', 
                        palette='viridis')
            plt.title('Model Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
        if plot_scatter:
            plt.figure(figsize=(8, 6))  # Set figure size
            plt.scatter(self.predictions, self.y_test, color='blue', alpha=0.5)  # Create scatter plot

            # Add labels and title
            plt.xlabel('predicted home team wining margin')
            plt.ylabel('actual wining margin')
            plt.title('Predicted wining margin vs Actual wining margin')

            # Display the plot
            plt.grid(True)  # Add grid
            plt.show()
            
        

        
        
        

