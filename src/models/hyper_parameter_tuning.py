import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, log_loss
from sklearn.preprocessing import StandardScaler
import warnings

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.xgboost_randomforrest_model_v2 import NFLModelV2
from src.config import config_v2
warnings.filterwarnings('ignore')

def add_game_week_qid(df, season_col='season', week_col='week', qid_col='qid'):
    """
    Add a query ID (qid) for each unique combination of season and week.
    All games in the same season and week will have the same qid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing NFL game data
    season_col : str, default 'season'
        Name of the column containing season information
    week_col : str, default 'week'
        Name of the column containing week information
    qid_col : str, default 'qid'
        Name of the new column to create for query IDs
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added qid column
        
    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'season': [2023, 2023, 2023, 2024, 2024, 2024],
    ...     'week': [1, 1, 2, 1, 1, 2],
    ...     'team': ['A', 'B', 'C', 'D', 'E', 'F']
    ... })
    >>> df_with_qid = add_game_week_qid(df)
    >>> print(df_with_qid)
       season  week team  qid
    0    2023     1    A    0
    1    2023     1    B    0
    2    2023     2    C    1
    3    2024     1    D    2
    4    2024     1    E    2
    5    2024     2    F    3
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Check if required columns exist
    if season_col not in df_copy.columns:
        raise ValueError(f"Column '{season_col}' not found in DataFrame")
    if week_col not in df_copy.columns:
        raise ValueError(f"Column '{week_col}' not found in DataFrame")
    
    # Create a unique identifier for each season-week combination
    # Sort by season and week to ensure chronological order
    unique_combinations = (df_copy[[season_col, week_col]]
                          .drop_duplicates()
                          .sort_values([season_col, week_col])
                          .reset_index(drop=True))
    
    # Assign sequential qid values
    unique_combinations[qid_col] = range(len(unique_combinations))
    
    # Merge back to original DataFrame
    df_with_qid = df_copy.merge(
        unique_combinations, 
        on=[season_col, week_col], 
        how='left'
    )
    
    # Ensure qid is integer type
    df_with_qid[qid_col] = df_with_qid[qid_col].astype(int)
    
    print(f"Added {qid_col} column with {df_with_qid[qid_col].nunique()} unique season-week combinations")
    print(f"QID range: {df_with_qid[qid_col].min()} to {df_with_qid[qid_col].max()}")
    
    return df_with_qid

class NFLHyperparameterTuner:
    def __init__(self, model_obj, target_columns, date_column=None):
        """
        Initialize the hyperparameter tuner
        
        Parameters:
        -----------
        data : pd.DataFrame
            Your NFL dataset
        target_columns : dict
            Dictionary with target column names as keys and task types as values
            e.g., {'binary_over_under': 'classification', 'binary_spread': 'classification',
                   'score_diff': 'regression', 'total_points': 'regression'}
        feature_columns : list
            List of feature column names
        date_column : str, optional
            Date column name for proper time series sorting
        """
        self.data=add_game_week_qid(model_obj._dataset.copy())
        self.target_columns = target_columns
        self.feature_columns = model_obj.feature_columns
        self.date_column = date_column
        
        # Sort by date if date column is provided
        
        self.X = self.data[self.feature_columns]
        self.scalers = {}
        self.best_params = {}
        self.studies = {}
        


    
    def objective_function(self, trial, target_name, task_type):
        """
        Objective function for Optuna optimization
        """
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, ),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 10),
            "n_estimators":trial.suggest_int('n_estimators', 100, 5000),
            "early_stopping_rounds":trial.suggest_int('early_stopping_rounds', 10, 1000),
            "lookback_seasons": trial.suggest_int('lookback_seasons', 2,5)
            
        }
        
        # Number of boosting rounds
        n_estimators = params.pop("n_estimators")
        
        # Early stopping rounds
        early_stopping_rounds = params.pop("early_stopping_rounds")
        
        lookback_seasons=params.pop("lookback_seasons")
        
        # Set objective based on task type
        if task_type == 'classification':
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
        else:  # regression
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        y = self.data[target_name].values
        
        for train_idx, test_idx in tscv.split(self.data[self.date_column]):
            train_idx_unique=np.unique(self.data.iloc[train_idx]["qid"])
            train_idx_int=int(train_idx_unique.shape[0]*.2)
            train_qids=train_idx_unique[:-train_idx_int]
            val_qids=train_idx_unique[-train_idx_int:]
            training_data=self.data[self.data["qid"].isin(train_qids)]
            
            val_data=self.data[self.data["qid"].isin(val_qids)]
            current_season=np.max(val_data["season"])
            lookback_season=current_season-lookback_seasons
            
            if lookback_season>config_v2.model_config_v2.start_season:
            
                training_data=training_data.iloc[np.where(training_data["season"]>=lookback_season)]
            train_idx_unique=np.unique(self.data.iloc[test_idx]["qid"])
            testing_data=self.data[self.data["qid"].isin(train_idx_unique)]
            
            y_test=testing_data[target_name].values
            
            # Scale features
            if task_type=="classification":
                neg_count = np.sum(training_data[target_name].values == 0)
                pos_count = np.sum(training_data[target_name].values == 1)
                scale_pos_weight = neg_count / pos_count
                params["scale_pos_weight"]=scale_pos_weight
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(training_data[self.feature_columns].values, label=training_data[target_name].values)
            dval = xgb.DMatrix(val_data[self.feature_columns].values, label=val_data[target_name].values)
            dtest=xgb.DMatrix(testing_data[self.feature_columns].values, label=y_test)
            
            # Train model with early stopping
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'validation')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # Make predictions
            y_pred = model.predict(dtest)
            
            # Calculate score based on task type
            if task_type == 'classification':
                # Use AUC as the metric for classification
                # score = -log_loss(y_test, y_pred)
                score=roc_auc_score(y_test, y_pred)
                
                
            else:  # regression
                # Use negative RMSE as the metric for regression (Optuna maximizes)
                score = -np.sqrt(mean_squared_error(y_test, y_pred))
            
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def tune_hyperparameters(self, n_trials=100, n_jobs=-1):
        """
        Tune hyperparameters for all targets
        """
        for target_name, task_type in self.target_columns.items():
            print(f"\nTuning hyperparameters for {target_name} ({task_type})...")
            
            # Create study
            direction = 'maximize'  # We want to maximize AUC for classification and minimize RMSE (but we negate it)
            study = optuna.create_study(direction=direction)
            
            # Optimize
            study.optimize(
                lambda trial: self.objective_function(trial, target_name, task_type),
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
            # Store results
            self.studies[target_name] = study
            self.best_params[target_name] = study.best_params
            
            print(f"Best score for {target_name}: {study.best_value:.4f}")
            print(f"Best parameters for {target_name}:")
            for param, value in study.best_params.items():
                print(f"  {param}: {value}")
    
    def train_final_models(self, test_size=config_v2.ModelConfigV2.test_size):
        """
        Train final models using best parameters
        """
        final_models = {}
        
        # Calculate split point for train/test
        split_point = int(len(self.data) * (1 - test_size))
        
        for target_name, task_type in self.target_columns.items():
            print(f"\nTraining final model for {target_name}...")
            
            # Get best parameters
            best_params = self.best_params[target_name].copy()
            n_estimators = best_params.pop('n_estimators')
            early_stopping_rounds = best_params.pop('early_stopping_rounds')
            
            # Set objective and eval metric
            if task_type == 'classification':
                best_params['objective'] = 'binary:logistic'
                best_params['eval_metric'] = 'auc'
            else:
                best_params['objective'] = 'reg:squarederror'
                best_params['eval_metric'] = 'rmse'
            
            # Split data
            X_train = self.X.iloc[:split_point]
            X_test = self.X.iloc[split_point:]
            y_train = self.data[target_name].iloc[:split_point].values
            y_test = self.data[target_name].iloc[split_point:].values
            
            # Scale features
            X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
            self.scalers[target_name] = scaler
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            # Train final model
            model = xgb.train(
                params=best_params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # Make predictions and evaluate
            y_pred = model.predict(dtest)
            
            if task_type == 'classification':
                auc_score = roc_auc_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
                print(f"  Test AUC: {auc_score:.4f}")
                print(f"  Test Accuracy: {accuracy:.4f}")
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                print(f"  Test RMSE: {rmse:.4f}")
                print(f"  Test MAE: {mae:.4f}")
            
            final_models[target_name] = {
                'model': model,
                'scaler': scaler,
                'task_type': task_type,
                'best_params': best_params
            }
        
        return final_models
    
    def get_feature_importance(self, models):
        """
        Get feature importance for all models
        """
        feature_importance = {}
        
        for target_name, model_info in models.items():
            model = model_info['model']
            importance_dict = model.get_score(importance_type='weight')
            
            # Convert to DataFrame for easier viewing
            importance_df = pd.DataFrame([
                {'feature': f'f{i}', 'feature_name': self.feature_columns[i], 'importance': importance_dict.get(f'f{i}', 0)}
                for i in range(len(self.feature_columns))
            ]).sort_values('importance', ascending=False)
            
            feature_importance[target_name] = importance_df
            
            print(f"\nTop 10 features for {target_name}:")
            print(importance_df.head(10)[['feature_name', 'importance']])
        
        return feature_importance
    
    def save_results(self, models, filename_prefix="nfl_model"):
        """
        Save models and results
        """
        import pickle
        
        results = {
            'models': models,
            'best_params': self.best_params,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'scalers': self.scalers
        }
        
        with open(f'{filename_prefix}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to {filename_prefix}_results.pkl")
    
    def predict(self, models, new_data):
        """
        Make predictions on new data
        """
        predictions = {}
        
        for target_name, model_info in models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Scale features
            X_scaled = scaler.transform(new_data[self.feature_columns])
            
            # Make predictions
            dmatrix = xgb.DMatrix(X_scaled)
            pred = model.predict(dmatrix)
            
            predictions[target_name] = pred
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Load your data
    # data = pd.read_csv('your_nfl_data.csv')
    
    # Define your targets and their types

    
    # Define your feature columns
    feature_columns = [
        # Add your feature column names here
        # e.g., 'home_team_rating', 'away_team_rating', 'weather_temp', etc.
    ]
    
    # Initialize tuner
    # tuner = NFLHyperparameterTuner(
    #     data=data,
    #     target_columns=target_columns,
    #     feature_columns=feature_columns,
    #     date_column='game_date'  # optional
    # )
    
    # Tune hyperparameters
    # tuner.tune_hyperparameters(n_trials=100, n_jobs=-1)
    
    # Train final models
    # final_models = tuner.train_final_models(test_size=0.2)
    
    # Get feature importance
    # feature_importance = tuner.get_feature_importance(final_models)
    
    # Save results
    # tuner.save_results(final_models, "nfl_prediction_models")
    
    # Make predictions on new data
    # predictions = tuner.predict(final_models, new_data)
    
    print("NFL Hyperparameter Tuning Script Ready!")
    print("Uncomment the example usage section and provide your data to run.")
    
