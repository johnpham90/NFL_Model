import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any, Iterable
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = XGBClassifier = None
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import BaseCrossValidator
import pickle

from src.utils.db_utils import execute_query
from src.features.team_level_features.team_level_features_v2 import (
    list_parsed_stats,
)
from src.config.config_v2 import team_feature_configs_v2

TargetType = Literal["spread","total_points","binary_spread_label","binary_ou_label"]

@dataclass
class ModelArtifacts:
    model: object
    feature_columns: List[str]
    target: str
    metrics: Dict[str, float]
    model_type: str
    params: Dict[str, Any]
    feature_hash: str

class NFLModelV2:
    def __init__(self,
                 target: TargetType = "spread",
                 test_size: float = 0.2,
                 random_state: int = 42,
                 stats: Optional[List[str]] = None) -> None:
        # Basic config
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        # Build canonical stat list from config (legacy names -> canonical underscore)
        legacy_map = getattr(team_feature_configs_v2, "legacy_to_canonical", {}) or {}
        default_legacy = team_feature_configs_v2.all_features()
        def canon(x: str) -> str:
            return legacy_map.get(x, x.strip().lower().replace(" ", "_"))
        self.stats = [canon(s) for s in (stats or default_legacy)]
        # Placeholders
        self._dataset: Optional[pd.DataFrame] = None
        self.artifacts: Optional[ModelArtifacts] = None

    # ---------- Data & Features ----------
    def load_games(self, start_season: int):
        """Load per-team rows, expand composite columns, derive efficiencies.

        Uses explicit column list to avoid duplicate columns from ts.* and prevents
        mixed-type comparison issues when building defensive opponent id.
        """
        q = f"""
        SELECT
            g.gamesummaryid,
            g.season,
            g.week,
            g.hometeamid,
            g.awayteamid,
            g.homescore,
            g.awayscore,
            ts.teamid,
            ts.time_of_possession,
            ts.cmp_att_yd_td_int,
            ts.fourth_down_conv,
            ts.rush_yds_tds,
            ts.fumbles_lost,
            ts.turnovers,
            ts.sacked_yards,
            ts.third_down_conv,
            ts.total_yards,
            ts.first_downs,
            ts.penalties_yards
        FROM stats.gamesummary g
        JOIN stats.teamstats ts ON ts.gamesummaryid = g.gamesummaryid
        WHERE g.season >= {start_season}
        ORDER BY g.season, g.week, g.gamesummaryid;
        """
        df = execute_query(q)
        if df is None or df.empty:
            raise ValueError("No game data returned.")
        df = df.copy()
        parse_map = getattr(team_feature_configs_v2, "parse_data_colums", {}) or {}
        legacy_map = getattr(team_feature_configs_v2, "legacy_to_canonical", {}) or {}
        eff_specs = getattr(team_feature_configs_v2, "canonical_efficiency_specs", {}) or {}

        def canon(x: str) -> str:  # legacy -> canonical underscore
            return legacy_map.get(x, x.strip().lower().replace(" ", "_"))

        # Fix malformed composite (observed legacy)
        if "rush_yds_tds" in df.columns:
            df.loc[df["rush_yds_tds"] == "8--1-0", "rush_yds_tds"] = "8-1-0"

        # Expand composites
        for comp_col, parts in parse_map.items():
            if comp_col not in df.columns:
                continue
            toks = df[comp_col].fillna("").astype(str).str.split("-", expand=True)
            if toks.shape[1] < len(parts):
                continue
            for i, legacy_part in enumerate(parts):
                out_col = canon(legacy_part)
                df[out_col] = pd.to_numeric(toks[i], errors="coerce")

        # Time of possession to seconds
        if "time_of_possession" in df.columns:
            tp = df["time_of_possession"].fillna("").astype(str)
            mins = pd.to_numeric(tp.str.split(":").str[0], errors="coerce")
            secs = pd.to_numeric(tp.str.split(":").str[1], errors="coerce")
            df["time_of_possession_seconds"] = mins * 60 + secs

        # Efficiency stats
        for eff_name, (num_col, den_col) in eff_specs.items():
            if num_col in df.columns and den_col in df.columns:
                num = pd.to_numeric(df[num_col], errors="coerce")
                den = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
                df[eff_name] = num / den

        # Ensure ids are strings to avoid mixed-type comparisons
        for c in ("teamid","hometeamid","awayteamid"):
            if c in df.columns:
                df[c] = df[c].astype(str)

        # Defensive opponent id (vectorized compare on aligned arrays)
        df["def_teamid"] = np.where(df["teamid"].values == df["hometeamid"].values,
                                    df["awayteamid"].values,
                                    df["hometeamid"].values)

        df["week"] = pd.to_numeric(df["week"], errors="coerce")

        # Targets
        df["spread"] = df["homescore"] - df["awayscore"]
        df["total_points"] = df["homescore"] + df["awayscore"]
        if {"spreadfavoriteteam", "spreadteamcovered"}.issubset(df.columns):
            df["binary_spread_label"] = (df["spreadfavoriteteam"] == df["spreadteamcovered"]).astype(int)
        else:
            df["binary_spread_label"] = (df["spread"] > 0).astype(int)
        if "overunderresults" in df.columns:
            df["binary_ou_label"] = (df["overunderresults"] == "Over").astype(int)
        else:
            df["binary_ou_label"] = (df["total_points"] > df["total_points"].median()).astype(int)

        # Per-team points (offensive points scored by this team in the game)
        df["team_points"] = np.where(
            df["teamid"] == df["hometeamid"], df["homescore"], df["awayscore"]
        ).astype("float32")
        if "team_points" not in self.stats:
            self.stats.append("team_points")

        self.games = df

    def build_feature_matrices(self, exclude_current: bool = True, include_defense: bool = True, include_differentials: bool = True):
        if not hasattr(self, "games"):
            raise RuntimeError("Call load_games first.")
        df = self.games.sort_values(["season","week","gamesummaryid"]).copy()
        available = set(df.columns)
        roll_sizes = getattr(team_feature_configs_v2, 'rolling_windows', [2,5,10])
        feature_frames: List[pd.DataFrame] = []
        built: List[str] = []
        skipped: List[str] = []
        # Pre-grouped index objects for efficiency
        grp_week = df.groupby(["season","week"])
        for stat in self.stats:
            if stat not in available:
                skipped.append(stat)
                continue
            stat_series = pd.to_numeric(df[stat], errors="coerce")
            # Offensive expanding mean
            off_hist = (df.groupby("teamid")[stat]
                          .expanding()
                          .mean()
                          .shift(1 if exclude_current else 0)
                          .reset_index(level=0, drop=True))
            cols: Dict[str, pd.Series] = {f"{stat}__sg": stat_series,
                                          f"{stat}__off_hist": off_hist}
            if include_defense:
                def_hist = (df.groupby("def_teamid")[stat]
                              .expanding()
                              .mean()
                              .shift(1 if exclude_current else 0)
                              .reset_index(level=0, drop=True))
                cols[f"{stat}__def_hist"] = def_hist
                if include_differentials:
                    cols[f"{stat}__diff_hist"] = off_hist - def_hist
                # Rolling windows + differentials
                for w in roll_sizes:
                    off_roll = (df.groupby("teamid")[stat]
                                  .rolling(window=w, min_periods=1)
                                  .mean()
                                  .shift(1 if exclude_current else 0)
                                  .reset_index(level=0, drop=True))
                    def_roll = (df.groupby("def_teamid")[stat]
                                  .rolling(window=w, min_periods=1)
                                  .mean()
                                  .shift(1 if exclude_current else 0)
                                  .reset_index(level=0, drop=True))
                    cols[f"{stat}__off_roll{w}"] = off_roll
                    cols[f"{stat}__def_roll{w}"] = def_roll
                    if include_differentials:
                        cols[f"{stat}__diff_roll{w}"] = off_roll - def_roll
                # SOS ratios
                off_non0 = off_hist.replace(0, np.nan)
                def_non0 = def_hist.replace(0, np.nan)
                cols[f"{stat}__sos_ratio"] = off_non0 / def_non0
                cols[f"{stat}__sos_inv_ratio"] = def_non0 / off_non0
            # Z-scores (single game & off_hist) within week (avoid needing columns in df first)
            week_keys = [df["season"], df["week"]]
            sg_mean = stat_series.groupby(week_keys).transform('mean')
            sg_std = stat_series.groupby(week_keys).transform('std').replace(0, np.nan)
            off_mean = off_hist.groupby(week_keys).transform('mean')
            off_std = off_hist.groupby(week_keys).transform('std').replace(0, np.nan)
            cols[f"{stat}__sg_z"] = (stat_series - sg_mean) / sg_std
            cols[f"{stat}__off_hist_z"] = (off_hist - off_mean) / off_std
            feature_frames.append(pd.DataFrame(cols))
            built.extend(cols.keys())
        if skipped:
            print(f"[INFO] Skipped stats (missing in parsed data): {skipped}")
        # Concatenate all feature blocks at once (prevents fragmentation)
        if feature_frames:
            feature_block = pd.concat(feature_frames, axis=1)
            # Rebuild df including new features without repeated inserts
            df = pd.concat([df.reset_index(drop=True), feature_block.reset_index(drop=True)], axis=1)
        self._include_differentials = include_differentials and include_defense
        self._hist_features = built
        self._hist_df = df
        return built
    
    def build_dataset(self):
        if not hasattr(self, "_hist_df"):
            raise RuntimeError("Call build_feature_matrices first.")
        df = self._hist_df
        # Split into home / away team rows
        home = df[df["teamid"] == df["hometeamid"]].copy()
        away = df[df["teamid"] == df["awayteamid"]].copy()
        # Select only built feature columns
        feature_base = [c for c in self._hist_features if c in home.columns]
        home_core_cols = [
            "gamesummaryid","season","week","hometeamid","awayteamid","homescore","awayscore",
            "spread","total_points","binary_spread_label","binary_ou_label"
        ]
        home_sel = home[home_core_cols + feature_base]
        away_sel = away[["gamesummaryid"] + feature_base].rename(columns={c: f"opp_{c}" for c in feature_base})
        game_df = home_sel.merge(away_sel, on="gamesummaryid", how="left")
        self.feature_columns = feature_base + [f"opp_{c}" for c in feature_base]
        self._dataset = game_df.reset_index(drop=True)
        return self._dataset

    # ---------- Training ----------
    def _select_X_y(self):
        if self._dataset is None:
            raise RuntimeError("Dataset not built.")
        df = self._dataset.sort_values(["season","week"]).reset_index(drop=True)
        feature_cols = [c for c in self.feature_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not feature_cols:
            raise ValueError("No feature columns selected.")
        X = df[feature_cols].ffill().fillna(0)
        y = df[self.target]
        is_class = self.target in {"binary_spread_label","binary_ou_label"}
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test, feature_cols, is_class

    class _BlockedTimeSeriesCV(BaseCrossValidator):
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

    def _eval(self, y_true, y_pred, is_class: bool):
        if is_class:
            acc = accuracy_score(y_true, y_pred)
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
            return {"accuracy": acc, "precision": p, "recall": r, "f1": f}
        return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}

    def fit_random_forest(self, param_dist: Optional[Dict] = None):
        X_train, X_test, y_train, y_test, feature_cols, is_class = self._select_X_y()
        base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1) if is_class else RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        model_type = 'RandomForestClassifier' if is_class else 'RandomForestRegressor'
        if param_dist:
            cv = self._BlockedTimeSeriesCV(n_splits=5)
            search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=min(25, len(param_dist)*5), cv=cv, n_jobs=-1, random_state=self.random_state)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            used_params = search.best_params_
        else:
            model = base.fit(X_train, y_train)
            used_params = getattr(model, 'get_params', lambda: {})()
        preds = model.predict(X_test)
        metrics = self._eval(y_test, preds, is_class)
        feature_hash = hashlib.sha256(('|'.join(feature_cols)).encode()).hexdigest()[:16]
        self.artifacts = ModelArtifacts(model, feature_cols, self.target, metrics, model_type, used_params, feature_hash)
        return metrics

    def fit_xgb(self, param_dist: Optional[Dict] = None):
        if XGBRegressor is None:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        X_train, X_test, y_train, y_test, feature_cols, is_class = self._select_X_y()
        base = XGBClassifier(random_state=self.random_state, n_estimators=300, verbosity=0, eval_metric='logloss') if is_class else XGBRegressor(random_state=self.random_state, n_estimators=300, verbosity=0)
        model_type = 'XGBClassifier' if is_class else 'XGBRegressor'
        if param_dist:
            cv = self._BlockedTimeSeriesCV(n_splits=5)
            search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=min(25, len(param_dist)*5), cv=cv, n_jobs=-1, random_state=self.random_state)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            used_params = search.best_params_
        else:
            model = base.fit(X_train, y_train)
            used_params = getattr(model, 'get_params', lambda: {})()
        preds = model.predict(X_test)
        metrics = self._eval(y_test, preds, is_class)
        feature_hash = hashlib.sha256(('|'.join(feature_cols)).encode()).hexdigest()[:16]
        self.artifacts = ModelArtifacts(model, feature_cols, self.target, metrics, model_type, used_params, feature_hash)
        return metrics

    def predict_matchup(self, season: int, week: int, home_team: int, away_team: int):
        if not self.artifacts:
            raise RuntimeError("Model not trained.")
        # Build a single-row feature set for prediction using already computed expanding means
        if not hasattr(self, "_hist_df"):
            raise RuntimeError("Model features not built.")
        # Filter historical rows up to requested week
        hist = self._hist_df
        mask = (hist.season == season) & (hist.week == week)
        if mask.sum() == 0:
            raise ValueError("Requested season/week not in historical feature frame.")
        home_row = hist[mask & (hist.teamid == home_team)].tail(1)
        away_row = hist[mask & (hist.teamid == away_team)].tail(1)
        if home_row.empty or away_row.empty:
            raise ValueError("Missing team rows for prediction week.")
        feats = {}
        for c in self._hist_features:
            feats[c] = home_row.iloc[0][c] if c in home_row.columns else 0.0
            feats[f"opp_{c}"] = away_row.iloc[0][c] if c in away_row.columns else 0.0
        X = pd.DataFrame([feats]).reindex(columns=self.artifacts.feature_columns, fill_value=0.0)
        return self.artifacts.model.predict(X)[0]

    def save(self, path: str):
        if not self.artifacts:
            raise RuntimeError("Nothing to save.")
        with open(path, "wb") as f:
            pickle.dump(self.artifacts, f)

    @staticmethod
    def load(path: str) -> 'NFLModelV2':
        with open(path, "rb") as f:
            art: ModelArtifacts = pickle.load(f)
        inst = NFLModelV2(target=art.target)
        inst.artifacts = art
        return inst