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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, r2_score
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.nfl_model import NFLModelV2, ModelArtifacts
from src.config import config
from src.models import hyper_parameter_tuning
import pickle
from src.evaluation import model_analysis
import numpy as np
# List of targets to train. Adjust ordering or remove as needed.
TARGETS = config.TARGETS

for tgt, i_type in config.models.items():

    model = NFLModelV2(target=tgt)\
        
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
    tuner.tune_model_hyperparameters(n_trials=50, target_name=tgt, task_type=i_type)
    
    params=tuner.best_params.copy()
    
    season_lookback=params.pop("lookback_seasons")
    
    start_season=2025-season_lookback
    model = NFLModelV2(target=tgt)
    model.load_games(start_season=start_season)
    # Build offensive + defensive + differential features
    model.build_feature_matrices(include_defense=True, include_differentials=True)
    model.build_dataset()
    
    n_estimators = params.pop("n_estimators")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    
    # Set objective and metrics based on task type
    if i_type == "classification":
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
    else:
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
    
    X_train, X_test, y_train_, y_test, feature_cols, is_class = model._select_X_y()
    
    dtrain = xgb.DMatrix(X_train.values, label=y_train_.values, feature_names=X_train.columns.tolist())
    dtest = xgb.DMatrix(X_test.values, y_test.values, feature_names=X_train.columns.tolist())

    # scale_pos_weight only for classification
    if i_type == "classification":
        neg_count = np.sum(y_train_.values == 0)
        pos_count = np.sum(y_train_.values == 1)
        scale_pos_weight = neg_count / pos_count
        params["scale_pos_weight"] = scale_pos_weight
    
    # Train model with early stopping
    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=1
    )
    predictions = xgb_model.predict(dtest)
    
    # Calculate metrics based on task type
    if i_type == "classification":
        preds = (predictions >= 0.5).astype(int)
        acc = accuracy_score(y_test.values, preds)
        p, r, f, _ = precision_recall_fscore_support(y_test.values, preds, average="weighted", zero_division=0)
        metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f}
        model_type_str = 'XGBClassifier'
    else:
        preds = predictions
        mae = mean_absolute_error(y_test.values, preds)
        r2 = r2_score(y_test.values, preds)
        metrics = {"mae": mae, "r2": r2}
        model_type_str = 'XGBRegressor'
    
    # Create proper ModelArtifacts wrapper (required by predictions.py)
    feature_hash = hashlib.sha256(('|'.join(feature_cols)).encode()).hexdigest()[:16]
    artifact = ModelArtifacts(
        model=xgb_model,
        feature_columns=feature_cols,
        target=tgt,
        metrics=metrics,
        model_type=model_type_str,
        params=params,
        feature_hash=feature_hash
    )
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    out_path = artifacts_dir / f"{tgt}_xgb_v2_{ts}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(artifact, f)
        
    

    print(f"[DONE] {tgt}: {out_path}")

    # Run analysis only for classification targets
    if i_type == "classification":
        win_loss_analyzer = model_analysis.NFLModelAnalyzer(
            model_type='classification',
            model_name=tgt,
            target_names=['Loss', 'Win'])
        
        binary_predictions = np.where(predictions >= 0.5, 1, 0)
        win_loss_analyzer.set_predictions(y_test.values, binary_predictions, predictions)
        win_loss_analyzer.full_analysis()
    else:
        print(f"  Metrics: MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}")