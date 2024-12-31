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

from src.features import team_level_features_class
from src import config

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
        self.create_spread_dict()
        self.random_state=42
        
        
        
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

                    df.loc[f"{i_game_week}_{i_game}", f"home team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][0]]
                    df.loc[f"{i_game_week}_{i_game}", f"away team {i_feature}"]=current_feature_data.loc[i_game_week, self.spread_dict[i_game_week][1]]
                    df.loc[f"{i_game_week}_{i_game}", "spread"]=self.spread_dict[i_game_week][2]                       

        self.x_y=df
        
    
    def build_random_forest_model(self, param_distirbution):
        df=self.x_y.dropna()
        
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        
        self.x_train, self.x_test, self.y_train, self.y_test=train_test_split(x,y, test_size=config.model_config.train_test_split, shuffle=False)
        
        rf_model=RandomForestRegressor(random_state=self.random_state)
        
        model_parameters=RandomizedSearchCV(rf_model, param_distirbution,n_jobs=-1,cv=5).fit(self.x_train, self.y_train).best_params_
        
        self.rf_model=RandomForestRegressor(**model_parameters).fit(X=self.x_train, y=self.y_train)
    def build_xg_boost_model(self, param_distirbution):
        df=self.x_y.dropna()
        
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        
        self.x_train, self.x_test, self.y_train, self.y_test=train_test_split(x,y, test_size=config.model_config.train_test_split, shuffle=False)
        
        xg_model=XGBRegressor(random_state=self.random_state)
        
        model_parameters=RandomizedSearchCV(xg_model, param_distirbution,n_jobs=-1,cv=5).fit(self.x_train.to_numpy(), self.y_train.to_numpy()).best_params_
        
        self.xg_model=XGBRegressor(**model_parameters).fit(X=self.x_train.to_numpy(), y=self.y_train.to_numpy())
    
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
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', 
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
            
        

        
        
        

