from pathlib import Path
from datetime import datetime


import sys
import os
current_dir = os.getcwd()
project_root = os.path.join(current_dir, '..', '..') 

sys.path.append(project_root)
from src.models.nfl_model import NFLModelV2
from src.models import hyper_parameter_tuning
from src.config import config
from src.evaluation import model_analysis
import numpy as np

import xgboost as xgb
import pickle


TARGETS = config.models


model = NFLModelV2()
model.load_games(start_season=config.model_config_v2.start_season)
# Build offensive + defensive + differential features
model.build_feature_matrices(include_defense=True, include_differentials=True)
model.build_dataset()



tuner = hyper_parameter_tuning.NFLHyperparameterTuner(
    model,
    target_columns=TARGETS,
    date_column='qid'  # optional
)


tuner.tune_hyperparameters(n_trials=20, n_jobs=-1)



models={}
best_params_dict=tuner.best_params.items()
for i_label, i_param in best_params_dict:
# for i_label, i_param in tuner.best_params['binary_ou_label',"total_points"]:
    params=i_param.copy()
    print(i_label, i_param)
    season_lookback=params.pop("lookback_seasons")
    start_season=2025-season_lookback
    model = NFLModelV2(target=i_label)
    model.load_games(start_season=start_season)
    # Build offensive + defensive + differential features
    model.build_feature_matrices(include_defense=True, include_differentials=True)
    model.build_dataset()
    
    n_estimators = params.pop("n_estimators")
        
        # Early stopping rounds
    early_stopping_rounds = params.pop("early_stopping_rounds")
    task_type=config.models[i_label]
    if task_type == 'classification':
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
    
    
    X_train, X_test, y_train_, y_test, feature_cols, is_class = model._select_X_y()
    
    if task_type == 'classification':
        neg_count = np.sum(y_train_.values == 0)
        pos_count = np.sum(y_train_.values == 1)
        scale_pos_weight = neg_count / pos_count
        params["scale_pos_weight"]=scale_pos_weight
    
    x_train=X_train.iloc[:int(-X_train.shape[0]*config.model_config_v2.test_size)]
    x_val=X_train.iloc[int(-X_train.shape[0]*config.model_config_v2.test_size):]
    y_train=y_train_.iloc[:int(-X_train.shape[0]*config.model_config_v2.test_size)]
    y_val=y_train_.iloc[int(-X_train.shape[0]*config.model_config_v2.test_size):]
        
    dtrain = xgb.DMatrix(x_train.values, label=y_train.values)
    dval = xgb.DMatrix(x_val.values, label=y_val.values)
    dtest=xgb.DMatrix(X_test.values, y_test.values)

    
            # Train model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=1
    )

    
    models[i_label]=model
    predictions=model.predict(dtest)
    
    if task_type == 'classification':
        win_loss_analyzer = model_analysis.evaluationNFLModelAnalyzer(
    model_type='classification',
    model_name=i_label,
    target_names=['Loss', 'Win']
    
)
        try:
            binary_predictions=np.where(predictions>=.5, 1,0)
            win_loss_analyzer.set_predictions(y_test.values, binary_predictions, predictions)
            win_loss_analyzer.full_analysis()
        except:
            pass
    else:
        score_analyzer = model_analysis.NFLModelAnalyzer(
            model_type='regression',
            model_name=i_label
        )
        score_analyzer.set_predictions(y_test.values, predictions)
        score_analyzer.full_analysis()