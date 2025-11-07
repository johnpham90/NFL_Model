"""
Train models for all supported targets in one run.

Targets:
- spread:       Home score minus away score (regression)
- total_points: Combined game points (regression)
- binary_spread_label: 1 if favorite covered (classification)
- binary_ou_label:     1 if game went over (classification)

Outputs:
- One pickled model artifact per target under artifacts/
  Named: <target>_rf_v2_<timestamp>.pkl
"""

from pathlib import Path
from datetime import datetime
import sys
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.xgboost_randomforrest_model_v2 import NFLModelV2
from src.config import config_v2
from src.models import hyper_parameter_tuning
import pickle
from src.evaluation import model_analysis
import numpy as np
# List of targets to train. Adjust ordering or remove as needed.
TARGETS = config_v2.TARGETS

for tgt, i_type in config_v2.models.items():

    model = NFLModelV2(target=tgt)
    model.load_games(start_season=2010)
    # Build offensive + defensive + differential features
    model.build_feature_matrices(include_defense=True, include_differentials=True)
    model.build_dataset()
    # Random Forest baseline (swap to fit_xgb for XGBoost)
    tuner = hyper_parameter_tuning.NFLHyperparameterTuner(
    model,
    target_columns=TARGETS,
    date_column='qid'  # optional
)
    tuner.tune_model_hyperparameters_rf(n_trials=50, target_name=tgt, task_type=i_type)
    
    params=tuner.best_params_rf.copy()
    params["class_weight"]="balanced_subsample"
    
    season_lookback=params.pop("lookback_seasons")
    
    start_season=2025-season_lookback
    model = NFLModelV2(target=tgt)
    model.load_games(start_season=start_season)
    # Build offensive + defensive + differential features
    model.build_feature_matrices(include_defense=True, include_differentials=True)
    model.build_dataset()
    

    

    
    X_train, X_test, y_train_, y_test, feature_cols, is_class = model._select_X_y()
    

        
    
            # Train model with early stopping
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_)
    predictions=model.predict_proba(X_test)

    ts = datetime.utcnow().strftime("%Y%m%d")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    out_path = artifacts_dir / f"{tgt}_rf_v2_{ts}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
        
    

    print(f"[DONE] {tgt}: {out_path}")

    win_loss_analyzer = model_analysis.NFLModelAnalyzer(
    model_type='classification',
    model_name=f"{tgt}_rf",
    target_names=['Loss', 'Win'])
    
    binary_predictions=np.where(predictions[:,1]>=.5, 1,0)
    win_loss_analyzer.set_predictions(y_test.values, binary_predictions, predictions[:,1])
    win_loss_analyzer.full_analysis()