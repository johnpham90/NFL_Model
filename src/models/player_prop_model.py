"""Player prop prediction model mirroring NFLModelV2 architecture."""
import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = XGBClassifier = None
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import BaseCrossValidator
import pickle

from src.config.player_prop_config import (
    player_prop_feature_config,
    PropType,
    PositionType,
    TargetType
)
from src.features.player_prop_features.main_player_prop_builder import build_player_features
from src.features.player_prop_features.player_prop_features import (
    validate_dataframe,
    validate_target_column
)


@dataclass
class PlayerPropArtifacts:
    """Storage for trained model and metadata."""
    model: object
    feature_columns: List[str]
    position: PositionType
    prop_type: PropType
    target_type: TargetType
    metrics: Dict[str, float]
    model_type: str
    params: Dict[str, Any]
    feature_hash: str


class NFLPlayerPropModel:
    """
    Player prop prediction model following NFLModelV2 architecture.
    Supports position-specific models for QB, RB, WR/TE props.
    
    Features:
    - Time-series rolling features (expanding and windowed means/stds)
    - Opponent defense aggregations
    - Player vs opponent matchup history
    - Proper temporal splits to prevent data leakage
    """
    
    def __init__(
        self,
        position: PositionType,
        prop_type: PropType,
        target_type: TargetType = "yards",
        test_size: float = 0.2,
        random_state: int = 42,
        stats: Optional[List[str]] = None
    ):
        """
        Initialize player prop model.
        
        Args:
            position: Player position (QB, RB, WR, TE, PASS_CATCHER)
            prop_type: Type of prop (pass_yds, rush_yds, rec_yds, receptions)
            target_type: What to predict (yards, td_probability, receptions)
            test_size: Proportion of data for test set (default: 0.2)
            random_state: Random seed for reproducibility
            stats: Optional list of stats to use (defaults to config for prop_type)
        """
        self.position = position
        self.prop_type = prop_type
        self.target_type = target_type
        self.test_size = test_size
        self.random_state = random_state
        
        # Get feature stats from config or use provided list
        if stats is None:
            self.stats = player_prop_feature_config.get_features_for_prop(prop_type)
        else:
            self.stats = stats
        
        # Internal state
        self._dataset: Optional[pd.DataFrame] = None
        self.artifacts: Optional[PlayerPropArtifacts] = None
        
        # Columns that should never be features (target leakage prevention)
        self._OUTCOME_COLS = {
            "pass_yds", "rush_yds", "rec_yds", "receptions",
            "pass_td", "rush_td", "rec_td", "rec", "targets", "has_any_td" 
        }
    
    # ========== DATA LOADING ==========
    
    def load_player_games(self, start_season: int) -> pd.DataFrame:
        """
        Load player game-level data using centralized feature builder.
        
        Uses the main builder to orchestrate all feature engineering:
        - Base stats loading
        - Position filtering  
        - Game lines integration
        - TD indicators
        
        Args:
            start_season: First season to include (e.g., 2020)
            
        Returns:
            DataFrame with player game records and all features
            
        Raises:
            ValueError: If prop_type is unknown or no data returned
        """
        print(f"Loading {self.prop_type} data for {self.position} from season {start_season}...")

        # Use the main builder - includes game lines automatically
        self.player_games = build_player_features(
            start_season=start_season,
            prop_type=self.prop_type,
            position=self.position,
            add_indicators=True,
            add_lines=True  # Include game line features
        )
        
        validate_dataframe(
            self.player_games, 
            min_rows=50, 
            context=f"{self.prop_type} for {self.position}"
        )
        
        print(f"Loaded {len(self.player_games)} player-game records")
        return self.player_games
    
    # ========== FEATURE ENGINEERING ==========
    
    def build_feature_matrices(
        self,
        exclude_current: bool = True,
        include_opponent_defense: bool = True,
        include_matchup_history: bool = True
    ) -> List[str]:
        """
        Build rolling features, opponent defense stats, and matchup history.
        
        Creates time-series features that respect temporal ordering to prevent
        data leakage. Features include:
        - Player historical averages (expanding mean)
        - Rolling window means and standard deviations
        - Z-scores relative to weekly averages
        - Opponent defense aggregations (optional)
        - Player vs opponent matchup history (optional)
        
        Args:
            exclude_current: If True, exclude current game from rolling stats
            include_opponent_defense: Add opponent defensive metrics
            include_matchup_history: Add player career stats vs opponent
            
        Returns:
            List of feature column names created
            
        Raises:
            RuntimeError: If load_player_games() hasn't been called
        """
        if not hasattr(self, "player_games"):
            raise RuntimeError("Call load_player_games() first.")
        
        # Validate data before processing
        validate_dataframe(self.player_games, min_rows=30, context="feature engineering")
        
        df = self.player_games.sort_values(
            ["season", "week", "gamesummaryid"]
        ).copy()
        available = set(df.columns)
        roll_sizes = player_prop_feature_config.rolling_windows
        
        feature_frames: List[pd.DataFrame] = []
        built: List[str] = []
        skipped: List[str] = []
        
        # Helper functions for proper temporal lag
        def prior_expanding(series: pd.Series) -> pd.Series:
            """Expanding mean excluding current game."""
            s = series.shift(1 if exclude_current else 0)
            return s.expanding(min_periods=1).mean()
        
        def prior_rolling(series: pd.Series, w: int) -> pd.Series:
            """Rolling window mean excluding current game."""
            s = series.shift(1 if exclude_current else 0)
            return s.rolling(window=w, min_periods=1).mean()
        
        def prior_std(series: pd.Series, w: int) -> pd.Series:
            """Rolling window standard deviation excluding current game."""
            s = series.shift(1 if exclude_current else 0)
            return s.rolling(window=w, min_periods=2).std()
        
        player_grp = df.groupby("playerid", group_keys=False)
        
        # Build rolling features for each stat
        for stat in self.stats:
            # Check if column exists in dataframe
            if stat not in available:
                skipped.append(f"{stat} (not in data)")
                continue
            
            # We CAN build temporal features from outcome columns
            # (they use historical data, so no leakage)
            # We'll sanitize raw outcome columns later in _sanitize_feature_columns
            
            # Expanding mean (season-to-date average)
            player_hist = player_grp[stat].apply(prior_expanding)
            cols: Dict[str, pd.Series] = {f"{stat}__player_hist": player_hist}
            
            # Rolling windows (configurable sizes)
            for w in roll_sizes:
                cols[f"{stat}__player_roll{w}"] = player_grp[stat].apply(
                    lambda s: prior_rolling(s, w)
                )
                cols[f"{stat}__player_std{w}"] = player_grp[stat].apply(
                    lambda s: prior_std(s, w)
                )
            
            # Historical z-score (player relative to weekly average)
            week_keys = [df["season"], df["week"]]
            player_mean = player_hist.groupby(week_keys).transform('mean')
            player_std_z = player_hist.groupby(week_keys).transform('std').replace(0, np.nan)
            cols[f"{stat}__player_hist_z"] = (player_hist - player_mean) / player_std_z
            
            feature_frames.append(pd.DataFrame(cols))
            built.extend(cols.keys())
        
        if skipped:
            print(f"[INFO] Skipped stats (missing or forbidden): {skipped}")
        
        if feature_frames:
            feature_block = pd.concat(feature_frames, axis=1)
            df = pd.concat(
                [df.reset_index(drop=True), feature_block.reset_index(drop=True)], 
                axis=1
            )
        
        self._hist_features = built
        self._hist_df = df
        
        # Add optional feature sets
        if include_opponent_defense:
            self._add_opponent_defense_features()
        if include_matchup_history:
            self._add_matchup_history()
        
        return built
    
    def _add_opponent_defense_features(self):
        """
        Add opponent defensive stats (yards allowed by position).
        
        Aggregates how many yards the opponent defense has allowed to this
        position group historically. Uses proper temporal ordering to prevent
        data leakage.
        """
        print("Adding opponent defense features...")
        df = self._hist_df.copy()
        
        # Select appropriate target column based on prop type
        if self.prop_type == "pass_yds":
            target_col, feature_name = 'pass_yds', 'opp_pass_yds_allowed_hist'
        elif self.prop_type == "rush_yds":
            target_col, feature_name = 'rush_yds', 'opp_rush_yds_allowed_hist'
        elif self.prop_type in ["rec_yds", "receptions"]:
            target_col, feature_name = 'rec_yds', 'opp_rec_yds_allowed_hist'
        else:
            return
        
        # Aggregate yards allowed by opponent per week
        def_agg = df.groupby(['opp_teamid', 'season', 'week'])[target_col].mean().reset_index()
        def_agg = def_agg.rename(columns={target_col: f'opp_{target_col}_allowed'})
        def_agg = def_agg.sort_values(['opp_teamid', 'season', 'week'])
        
        # Calculate historical average (excluding current week)
        def_agg[feature_name] = (
            def_agg.groupby('opp_teamid')[f'opp_{target_col}_allowed']
            .shift(1)
            .expanding(min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        
        # Merge back to main dataframe
        df = df.merge(
            def_agg[['opp_teamid', 'season', 'week', feature_name]],
            on=['opp_teamid', 'season', 'week'],
            how='left'
        )
        
        self._hist_features.append(feature_name)
        self._hist_df = df.fillna(0)
    
    def _add_matchup_history(self):
        """
        Add player career history vs specific opponent (vectorized).
        
        Calculates player's historical average performance against each specific
        opponent. Uses vectorized operations for optimal performance.
        """
        print("Adding matchup history features...")
        df = self._hist_df.copy()
        
        # Select target column based on prop type
        target_col_map = {
            "pass_yds": "pass_yds",
            "rush_yds": "rush_yds",
            "rec_yds": "rec_yds",
            "receptions": "rec"
        }
        target_col = target_col_map.get(self.prop_type)
        if not target_col:
            return
        
        # Sort for proper time ordering
        df = df.sort_values(['playerid', 'opp_teamid', 'season', 'week'])
        
        # Calculate expanding mean vs opponent (excluding current game)
        df['player_vs_opp_avg'] = (
            df.groupby(['playerid', 'opp_teamid'])[target_col]
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)
        )
        
        # Count of prior games vs opponent
        df['player_vs_opp_games'] = (
            df.groupby(['playerid', 'opp_teamid']).cumcount()
        )
        
        self._hist_features.extend(['player_vs_opp_avg', 'player_vs_opp_games'])
        self._hist_df = df
    
    def build_dataset(self) -> pd.DataFrame:
        """
        Build final dataset with all features and target column.
        
        Combines all engineered features with core metadata columns and
        the appropriate target variable for the specified target_type.
        
        Returns:
            DataFrame ready for training with features and target
            
        Raises:
            RuntimeError: If build_feature_matrices() hasn't been called
        """
        if not hasattr(self, "_hist_df"):
            raise RuntimeError("Call build_feature_matrices() first.")
        
        df = self._hist_df.copy()
        feature_base = [c for c in self._hist_features if c in df.columns]
        
        # Core metadata columns
        core_cols = [
            "gamesummaryid", "playerid", "player", "teamid",
            "season", "week", "is_home"
        ]
        
        # Add appropriate target column
        target_map = {
            ("yards", "pass_yds"): "pass_yds",
            ("yards", "rush_yds"): "rush_yds",
            ("yards", "rec_yds"): "rec_yds",
            ("td_probability", "pass_yds"): "has_pass_td",
            ("td_probability", "rush_yds"): "has_rush_td",
            ("td_probability", "rec_yds"): "has_rec_td",
            ("any_td_probability", "any_td"): "has_any_td", 
            ("receptions", "receptions"): "rec"
        }
        target_col = target_map.get((self.target_type, self.prop_type))
        if target_col:
            core_cols.append(target_col)
        
        # Clean and deduplicate feature columns
        self.feature_columns = sorted(set(feature_base))
        self._sanitize_feature_columns()
        
        self._dataset = df[core_cols + self.feature_columns].reset_index(drop=True)
        return self._dataset
    
    # ========== TRAINING ==========
    
    def _select_X_y(self):
        """
        Select features and target for model training.
        
        Performs final feature sanitization and temporal train/test split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_cols, is_classification)
            
        Raises:
            RuntimeError: If dataset hasn't been built
            ValueError: If no valid features or unknown target_type
        """
        if self._dataset is None:
            raise RuntimeError("Call build_dataset() first.")
        
        df = self._dataset.sort_values(["season", "week"]).reset_index(drop=True)
        self._sanitize_feature_columns()
        
        # Select only numeric features, excluding outcome columns
        feature_cols = [
            c for c in self.feature_columns
            if c in df.columns
            and c not in self._OUTCOME_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        if not feature_cols:
            raise ValueError("No valid feature columns selected.")
        
        X = df[feature_cols].ffill().fillna(0)
        
        # Select target based on target_type
        target_config = {
            ("yards", "pass_yds"): ("pass_yds", False),
            ("yards", "rush_yds"): ("rush_yds", False),
            ("yards", "rec_yds"): ("rec_yds", False),
            ("td_probability", "pass_yds"): ("has_pass_td", True),
            ("td_probability", "rush_yds"): ("has_rush_td", True),
            ("td_probability", "rec_yds"): ("has_rec_td", True),
            ("any_td_probability", "any_td"): ("has_any_td", True),
            ("receptions", "receptions"): ("rec", False)
        }
        
        target_info = target_config.get((self.target_type, self.prop_type))
        if not target_info:
            raise ValueError(f"Unknown target combination: {self.target_type}, {self.prop_type}")
        
        target_col, is_class = target_info
        
        # Validate target column exists
        validate_target_column(df, target_col)
        
        y = df[target_col]
        
        # Validate sufficient data for split
        if len(X) < 50:
            raise ValueError(
                f"Insufficient data for training: {len(X)} samples (minimum 50 required)"
            )
        
        # Time-series split (no shuffling)
        split_idx = int(len(X) * (1 - self.test_size))
        
        if split_idx < 10 or len(X) - split_idx < 10:
            raise ValueError(
                f"Insufficient data for train/test split. "
                f"Train: {split_idx}, Test: {len(X) - split_idx} (need at least 10 each)"
            )
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols, is_class
    
    def _sanitize_feature_columns(self):
        """
        Remove outcome/target columns from features to prevent data leakage.
        
        Only removes exact matches of outcome columns, not historical/rolling features.
        Historical features like 'rec_yds__player_hist' are valid predictors.
        """
        if not hasattr(self, 'feature_columns') or self.feature_columns is None:
            return
        
        original_count = len(self.feature_columns)
        
        # Only remove EXACT outcome columns (raw columns without __ suffix)
        # Keep derived temporal features (those with __ in the name)
        cleaned = [
            c for c in self.feature_columns 
            if c not in self._OUTCOME_COLS or '__' in c
        ]
        
        if len(cleaned) != original_count:
            removed = original_count - len(cleaned)
            print(f"[INFO] Sanitized features: removed {removed} direct outcome columns (kept temporal features)")
        
        self.feature_columns = sorted(dict.fromkeys(cleaned))
    
    class _BlockedTimeSeriesCV(BaseCrossValidator):
        """Time-series cross-validation splitter."""
        def __init__(self, n_splits: int = 5):
            self.n_splits = n_splits
        
        def split(self, X, y=None, groups=None):
            n = len(X)
            fold_sizes = [n // self.n_splits] * self.n_splits
            for i in range(n % self.n_splits):
                fold_sizes[i] += 1
            
            current = 0
            for fold_size in fold_sizes:
                stop = current + fold_size
                train_end = stop - 1
                if train_end <= 0:
                    current = stop
                    continue
                yield range(0, train_end), range(train_end, stop)
                current = stop
        
        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits
    
    def _eval(self, y_true, y_pred, is_class: bool) -> Dict[str, float]:
        """Evaluate model performance."""
        if is_class:
            acc = accuracy_score(y_true, y_pred)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            return {"accuracy": acc, "precision": p, "recall": r, "f1": f}
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}
    
    def fit_random_forest(self, param_dist: Optional[Dict] = None) -> Dict[str, float]:
        """Train Random Forest model with optional hyperparameter tuning."""
        X_train, X_test, y_train, y_test, feature_cols, is_class = self._select_X_y()
        
        base = (
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            if is_class else
            RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        )
        model_type = 'RandomForestClassifier' if is_class else 'RandomForestRegressor'
        
        if param_dist:
            print(f"Running hyperparameter search with {len(param_dist)} parameters...")
            cv = self._BlockedTimeSeriesCV(n_splits=5)
            search = RandomizedSearchCV(
                base, param_distributions=param_dist, n_iter=min(25, len(param_dist) * 5),
                cv=cv, n_jobs=-1, random_state=self.random_state, verbose=1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            used_params = search.best_params_
            print(f"Best parameters: {used_params}")
        else:
            print("Training with default parameters...")
            model = base.fit(X_train, y_train)
            used_params = getattr(model, 'get_params', lambda: {})()
        
        preds = model.predict(X_test)
        metrics = self._eval(y_test, preds, is_class)
        feature_hash = hashlib.sha256(('|'.join(feature_cols)).encode()).hexdigest()[:16]
        
        self.artifacts = PlayerPropArtifacts(
            model=model, feature_columns=feature_cols, position=self.position,
            prop_type=self.prop_type, target_type=self.target_type,
            metrics=metrics, model_type=model_type, params=used_params,
            feature_hash=feature_hash
        )
        
        print(f"\nModel trained successfully!")
        print(f"Test set metrics: {metrics}")
        return metrics
    
    def fit_xgb(self, param_dist: Optional[Dict] = None) -> Dict[str, float]:
        """Train XGBoost model with optional hyperparameter tuning."""
        if XGBRegressor is None:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        X_train, X_test, y_train, y_test, feature_cols, is_class = self._select_X_y()
        
        base = (
            XGBClassifier(random_state=self.random_state, n_estimators=300, verbosity=0, eval_metric='logloss')
            if is_class else
            XGBRegressor(random_state=self.random_state, n_estimators=300, verbosity=0)
        )
        model_type = 'XGBClassifier' if is_class else 'XGBRegressor'
        
        if param_dist:
            print(f"Running hyperparameter search with {len(param_dist)} parameters...")
            cv = self._BlockedTimeSeriesCV(n_splits=5)
            search = RandomizedSearchCV(
                base, param_distributions=param_dist, n_iter=min(25, len(param_dist) * 5),
                cv=cv, n_jobs=-1, random_state=self.random_state, verbose=1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            used_params = search.best_params_
            print(f"Best parameters: {used_params}")
        else:
            print("Training with default parameters...")
            model = base.fit(X_train, y_train)
            used_params = getattr(model, 'get_params', lambda: {})()
        
        preds = model.predict(X_test)
        metrics = self._eval(y_test, preds, is_class)
        feature_hash = hashlib.sha256(('|'.join(feature_cols)).encode()).hexdigest()[:16]
        
        self.artifacts = PlayerPropArtifacts(
            model=model, feature_columns=feature_cols, position=self.position,
            prop_type=self.prop_type, target_type=self.target_type,
            metrics=metrics, model_type=model_type, params=used_params,
            feature_hash=feature_hash
        )
        
        print(f"\nModel trained successfully!")
        print(f"Test set metrics: {metrics}")
        return metrics
    
    def save(self, path: str):
        """Save trained model artifacts to disk."""
        if not self.artifacts:
            raise RuntimeError("Nothing to save. Train the model first.")
        
        with open(path, "wb") as f:
            pickle.dump(self.artifacts, f)
        
        print(f"Model saved to {path}")
        print(f"  Type: {self.artifacts.model_type}")
        print(f"  Features: {len(self.artifacts.feature_columns)}")
        print(f"  Hash: {self.artifacts.feature_hash}")
    
    @staticmethod
    def load(path: str):
        """Load trained model artifacts from disk."""
        with open(path, "rb") as f:
            art = pickle.load(f)
        
        inst = NFLPlayerPropModel(
            position=art.position,
            prop_type=art.prop_type,
            target_type=art.target_type
        )
        inst.artifacts = art
        
        print(f"Loaded {art.model_type} model for {art.prop_type}")
        print(f"  Target: {art.target_type}")
        print(f"  Features: {len(art.feature_columns)}")
        print(f"  Test metrics: {art.metrics}")
        
        return inst