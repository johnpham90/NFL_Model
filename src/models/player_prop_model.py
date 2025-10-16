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
    Supports position-specific models for QB, RB, and PASS_CATCHER (RB+WR+TE) props.
    
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
    stats: Optional[List[str]] = None, 
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
        """Load player game-level data using centralized feature builder."""
        print(f"Loading {self.prop_type} data for {self.position} from season {start_season}...")

        # Use builder with its defaults
        self.player_games = build_player_features(
            start_season=start_season,
            prop_type=self.prop_type,
            position=self.position,
            add_indicators=True
            # All other features controlled by builder defaults
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
        
        # Add efficiency metrics calculated from HISTORICAL data (no leakage)
        print(f"[INFO] Calculating efficiency metrics from historical data...")
        efficiency_features = self._build_efficiency_features(df, player_grp, prior_expanding, prior_rolling)
        if efficiency_features is not None and not efficiency_features.empty:
            df = pd.concat([df.reset_index(drop=True), efficiency_features.reset_index(drop=True)], axis=1)
            built.extend(efficiency_features.columns.tolist())
        
        self._hist_features = built
        self._hist_df = df
        
        return built
    
    def _build_efficiency_features(
        self,
        df: pd.DataFrame,
        player_grp,
        prior_expanding,
        prior_rolling
    ) -> pd.DataFrame:
        """
        Calculate efficiency metrics from historical data (no data leakage).
        
        Creates ratios like completion%, yds/att, td_rate from PREVIOUS games only.
        Uses shift(1) to ensure we never see current game stats.
        
        Args:
            df: DataFrame with current stats
            player_grp: Grouped DataFrame by playerid
            prior_expanding: Function for expanding mean with lag
            prior_rolling: Function for rolling mean with lag
        
        Returns:
            DataFrame with efficiency features
        """
        efficiency_cols = {}
        
        # QB Efficiency Metrics
        if 'pass_att' in df.columns and 'pass_cmp' in df.columns:
            completion_pct = player_grp.apply(
                lambda g: (g['pass_cmp'].shift(1).expanding(min_periods=1).sum() / 
                          g['pass_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['completion_pct__hist'] = completion_pct
        
        if 'pass_yds' in df.columns and 'pass_att' in df.columns:
            yds_per_att = player_grp.apply(
                lambda g: (g['pass_yds'].shift(1).expanding(min_periods=1).sum() / 
                          g['pass_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['yds_per_att__hist'] = yds_per_att
        
        if 'pass_yds' in df.columns and 'pass_cmp' in df.columns:
            yds_per_cmp = player_grp.apply(
                lambda g: (g['pass_yds'].shift(1).expanding(min_periods=1).sum() / 
                          g['pass_cmp'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['yds_per_cmp__hist'] = yds_per_cmp
        
        if 'pass_td' in df.columns and 'pass_att' in df.columns:
            td_rate = player_grp.apply(
                lambda g: (g['pass_td'].shift(1).expanding(min_periods=1).sum() / 
                          g['pass_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['td_rate__hist'] = td_rate
        
        if 'pass_int' in df.columns and 'pass_att' in df.columns:
            int_rate = player_grp.apply(
                lambda g: (g['pass_int'].shift(1).expanding(min_periods=1).sum() / 
                          g['pass_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['int_rate__hist'] = int_rate
        
        # RB Efficiency Metrics
        if 'rush_yds' in df.columns and 'rush_att' in df.columns:
            yds_per_carry = player_grp.apply(
                lambda g: (g['rush_yds'].shift(1).expanding(min_periods=1).sum() / 
                          g['rush_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['yds_per_carry__hist'] = yds_per_carry
        
        if 'rush_td' in df.columns and 'rush_att' in df.columns:
            rush_td_rate = player_grp.apply(
                lambda g: (g['rush_td'].shift(1).expanding(min_periods=1).sum() / 
                          g['rush_att'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['rush_td_rate__hist'] = rush_td_rate
        
        # WR/TE Efficiency Metrics
        if 'rec_yds' in df.columns and 'rec' in df.columns:
            yds_per_rec = player_grp.apply(
                lambda g: (g['rec_yds'].shift(1).expanding(min_periods=1).sum() / 
                          g['rec'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['yds_per_rec__hist'] = yds_per_rec
        
        if 'rec_yds' in df.columns and 'targets' in df.columns:
            yds_per_target = player_grp.apply(
                lambda g: (g['rec_yds'].shift(1).expanding(min_periods=1).sum() / 
                          g['targets'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['yds_per_target__hist'] = yds_per_target
        
        if 'rec' in df.columns and 'targets' in df.columns:
            catch_rate = player_grp.apply(
                lambda g: (g['rec'].shift(1).expanding(min_periods=1).sum() / 
                          g['targets'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['catch_rate__hist'] = catch_rate
        
        if 'rec_td' in df.columns and 'rec' in df.columns:
            rec_td_rate = player_grp.apply(
                lambda g: (g['rec_td'].shift(1).expanding(min_periods=1).sum() / 
                          g['rec'].shift(1).expanding(min_periods=1).sum().replace(0, np.nan))
            ).reset_index(level=0, drop=True)
            efficiency_cols['rec_td_rate__hist'] = rec_td_rate
        
        if efficiency_cols:
            print(f"[INFO] Created {len(efficiency_cols)} historical efficiency metrics")
            return pd.DataFrame(efficiency_cols)
        else:
            return pd.DataFrame()
    
    
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
        
        # Start with rolling/historical features
        feature_base = [c for c in self._hist_features if c in df.columns]
        
        # Add other feature columns that aren't metadata, stats, or targets
        # (e.g., opponent defense features, game lines, matchup history)
        core_metadata = [
            "gamesummaryid", "playerid", "player", "teamid",
            "season", "week", "is_home", "position"
        ]
        
        # ALL current-game statistics (outcome variables)
        # These columns exist in the data but should NOT be used as features
        # (they're used to CREATE rolling features, but not used directly)
        current_game_stats = {
            # Passing stats
            "pass_yds", "pass_td", "pass_att", "pass_cmp", "pass_int",
            "pass_air_yds", "pass_yac", "pass_target_yds", "pass_first_down",
            "pass_sacked", "pass_pressured", "pass_blitzed", "pass_hits", "pass_hurried",
            "pass_rating", "pass_drops", "pass_spikes", "pass_throwaways",
            # Derived passing efficiency (calculated from current game)
            "pass_air_yds_per_att", "pass_air_yds_per_cmp", "pass_yac_per_cmp",
            "pass_tgt_yds_per_att", "pass_poor_throw_pct", "pass_drop_pct",
            "pass_first_down_pct", "pass_pressured_pct",
            "completion_pct", "yds_per_att", "yds_per_cmp", "td_rate", "int_rate",
            # QB rushing stats
            "qb_rush_att", "qb_rush_yds", "qb_rush_td", "qb_rush_yac",
            "qb_rush_broken_tackles", "qb_rush_first_down",
            # Rushing stats
            "rush_yds", "rush_att", "rush_td", "rush_yac", "rush_first_down",
            "rush_broken_tackles", "rush_scrambles",
            "yds_per_carry", "fumble_rate",
            # Receiving stats
            "rec_yds", "rec", "rec_td", "rec_air_yds", "rec_yac", "rec_first_down",
            "rec_broken_tackles", "rec_drops", "targets", "rec_adot",
            "rec_target_int", "rec_catchable_targets",
            "yds_per_rec", "yds_per_target", "catch_rate",
            # Position-specific derived stats
            "rb_target_vol", "rb_catch_efficiency", "rb_shallow_routes",
            "wr_air_yards", "wr_deep_routes", "wr_explosiveness",
            "te_yac", "te_first_downs", "te_intermediate",
            # Binary outcomes
            "has_pass_td", "has_rush_td", "has_rec_td", "has_any_td"
        }
        
        # Safe pre-game features (known before game starts)
        pregame_features = {
            "days_rest", "weeks_from_bye", "is_favorite", "is_home",
            "is_rb", "is_wr", "is_te",  # Position indicators
        }
        
        # Find columns that are valid features:
        # - Not metadata
        # - Not current-game stats  
        # - Not already in rolling features
        # - Known before game time (lines, defense, matchup) OR explicitly pre-game
        additional_features = [
            c for c in df.columns 
            if c not in core_metadata
            and c not in current_game_stats
            and c not in feature_base
            and not c.startswith("_")  # Skip private columns
            # Include: lines (spread/total), opponent features (opp_), matchup (vs_opp), pre-game
            and (c in pregame_features or 
                 any(x in c for x in ['spread', 'total', 'moneyline', 'implied', 
                                      'opp_', 'vs_opp', 'player_vs_opp']))
        ]
        
        if additional_features:
            print(f"[INFO] Adding {len(additional_features)} non-rolling features: {additional_features[:10]}{'...' if len(additional_features) > 10 else ''}")
            feature_base.extend(additional_features)
        
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