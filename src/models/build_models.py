import pandas as pd
import numpy as np
import os
import sys
notebook_dir = os.path.abspath(os.path.dirname(''))
project_root = os.path.dirname(notebook_dir)
sys.path.append(project_root)
import pandas as pd
from utils.db_utils import get_connection, execute_query
import xgboost as xgb

from features.team_level_features import team_level_features_class
from models import xgboost_randomforrest_model
from utils.y_variable import build_spread_variable

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from utils import config

nfl_model=xgboost_randomforrest_model.NFLPredictor()
nfl_model.build_x_y_variable()

x=nfl_model.df_x.iloc[:,:-1].dropna()
y=nfl_model.df_y.loc[x.index, "binary_spread_label"]

for i_model in range(len(config.model_config.classification_models)):
    label=config.model_config.classification_models[i_model]
    model_name=config.model_config.classification_models_names[i_model]
    x=nfl_model.df_x.iloc[:,:-1].dropna()
    y=nfl_model.df_y.loc[x.index, label]
    nfl_model.build_xg_boost_model(x,y,config.model_config.param_dist,model_name, xgb.XGBClassifier)
for i_model in range(len(config.model_config.regression_models)):
    label=config.model_config.regression_models[i_model]
    model_name=config.model_config.regression_models_names[i_model]
    x=nfl_model.df_x.iloc[:,:-1].dropna()
    y=nfl_model.df_y.loc[x.index, label]
    nfl_model.build_xg_boost_model(x,y,config.model_config.param_dist,model_name, xgb.XGBRegressor)