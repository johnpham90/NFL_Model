import argparse
from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np
from scipy.stats import norm

from src.models.xgboost_randomforrest_model_v2 import NFLModelV2
from src.config.config_v2 import model_config_v2
from src.utils.schedule_utility import get_current_week

ALL_TARGETS = ["spread", "total_points", "binary_spread_label", "binary_ou_label"]

def latest_artifact(target: str, artifacts_dir: Path) -> Path:
    files = sorted(artifacts_dir.glob(f"{target}_rf_v2_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No artifacts found for target '{target}' in {artifacts_dir}")
    return files[-1]

def load_artifact(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_team_history(model: NFLModelV2, season: int, week: int):
    games = model.games
    mask = (games["season"] < season) | ((games["season"] == season) & (games["week"] < week))
    hist_games = games[mask]
    if hist_games.empty:
        raise ValueError(f"No historical games before season={season} week={week}.")
    original = model.games
    model.games = hist_games
    model.build_feature_matrices(exclude_current=True, include_defense=True, include_differentials=True)
    model.build_dataset()
    ds = model._dataset.sort_values(["season", "week"]).groupby("hometeamid").tail(1).set_index("hometeamid")
    model.games = original
    return ds

def assemble_matchups(artifact, team_hist: pd.DataFrame, season: int, week: int, schedule_wk: pd.DataFrame) -> pd.DataFrame:
    print(">>> ENTERED assemble_matchups")
    rows = []
    for _, g in schedule_wk.iterrows():
        home = g["hometeamid"]
        away = g["awayteamid"]
        if home not in team_hist.index or away not in team_hist.index:
            continue
        home_feats = team_hist.loc[home]
        away_feats = team_hist.loc[away]
        opp_feats = away_feats.copy()
        opp_feats.index = ["opp_" + c for c in opp_feats.index]
        combined = pd.concat([home_feats, opp_feats])
        for base in home_feats.index:
            diff_col = f"{base}_diff"
            opp_col = f"opp_{base}"
            if diff_col in artifact.feature_columns and opp_col in opp_feats.index:
                combined[diff_col] = combined[base] - combined[opp_col]
        combined["season"] = season
        combined["week"] = week
        combined["hometeamid"] = home
        combined["awayteamid"] = away
        
        # Set market_line_home and market_home_margin using spreadfavoriteteam logic (FIXED)
        if "spread" in g and "spreadfavoriteteam" in g:
            # Normalize team codes for robust comparison
            favorite = str(g["spreadfavoriteteam"]).upper().strip()
            home_team = str(g["hometeamid"]).upper().strip()
            away_team = str(g["awayteamid"]).upper().strip()
            spread_value = float(g["spread"])
            
            print(f"[DEBUG] Processing game: {home_team} vs {away_team}")
            print(f"[DEBUG] Spread favorite: {favorite}, Spread value: {spread_value}")
            
            # The spread value in your data appears to be the line for the favorite team
            # We need to convert this to the home team's perspective
            if favorite == home_team:
                # Home team is favored
                # The spread value is the line for the home team (favorite)
                market_home_margin = spread_value  # Should be negative if home team favored
                print(f"[DEBUG] Home team {home_team} is favored by {abs(spread_value)}")
            elif favorite == away_team:
                # Away team is favored, so home team is underdog
                # Flip the sign: if away team has -6.5, home team gets +6.5
                market_home_margin = -spread_value
                print(f"[DEBUG] Away team {favorite} is favored by {abs(spread_value)}, home team {home_team} gets +{abs(spread_value)}")
            else:
                # No clear favorite match (shouldn't happen with clean data)
                market_home_margin = 0.0
                print(f"[DEBUG] No favorite match found, setting spread to 0")
            
            # Keep market_line_home identical to market_home_margin to maintain clear semantics
            market_line_home = market_home_margin
            combined["market_line_home"] = market_line_home
            combined["market_home_margin"] = market_home_margin
            
            print(f"[DEBUG] Final market_home_margin for {home_team}: {market_home_margin}")
            
        if "over_under" in g:
            combined["market_total"] = g["over_under"]
        
        # SAFETY PATCH: Remove duplicate keys by keeping only the first occurrence
        combined = combined[~combined.index.duplicated(keep='first')]
        rows.append(combined)
    
    if not rows:
        raise ValueError("No matchup rows built (team IDs missing prior history?).")
    print(f"[DEBUG] Number of games processed in assemble_matchups: {len(rows)}")
    frame = pd.DataFrame(rows).set_index(["season", "week", "hometeamid", "awayteamid"])
    X = pd.DataFrame(columns=artifact.feature_columns)
    X = pd.concat([X, frame], axis=0)
    try:
        X = X[artifact.feature_columns].fillna(0)
    except Exception as e:
        print("[DEBUG] Exception during reindex:", e)
        raise
    return X.reset_index(), frame.reset_index()[["season","week","hometeamid","awayteamid"] + 
                                                [c for c in ["market_line_home","market_home_margin","market_total"]
                                                 if c in frame.columns]]
     

def win_probability(pred_spread, margin_std=13.5):
    return 1 - norm.cdf(0, loc=pred_spread, scale=margin_std)

def cover_probability(pred_spread, market_line, margin_std=13.5):
    return 1 - norm.cdf(market_line, loc=pred_spread, scale=margin_std)

def over_probability(pred_total, market_total, total_std=13.5):
    return 1 - norm.cdf(market_total, loc=pred_total, scale=total_std)

def predict_target(target: str,
                   season: int,
                   week: int,
                   start_season: int,
                   schedule_wk: pd.DataFrame,
                   artifacts_dir: Path,
                   output_dir: Path,
                   margin_std: float = 13.5,
                   total_std: float = 13.5) -> pd.DataFrame:
    print(f"[DEBUG] Entered predict_target for {target}, season={season}, week={week}")
    art_path = latest_artifact(target, artifacts_dir)
    artifact = load_artifact(art_path)
    model = NFLModelV2(target=target)
    model.load_games(start_season=start_season)
    team_hist = build_team_history(model, season=season, week=week)
    design_X, meta = assemble_matchups(artifact, team_hist, season, week, schedule_wk)
    X = design_X[artifact.feature_columns]
    preds = artifact.model.predict(X)
    out = meta.copy()
    out["target"] = target
    out["prediction"] = preds

    # Add win/cover/over probabilities for regression targets
    if target == "spread":
        out["win_prob"] = win_probability(preds, margin_std)
        if "market_line_home" in out:
            out["cover_prob"] = cover_probability(preds, out["market_line_home"], margin_std)
    if target == "total_points":
        if "market_total" in out:
            out["over_prob"] = over_probability(preds, out["market_total"], total_std)
    # Probabilities for classification targets
    if target.startswith("binary") and hasattr(artifact.model, "predict_proba"):
        probs = artifact.model.predict_proba(X)
        if probs.shape[1] == 2:
            out["prob_0"] = probs[:, 0]
            out["prob_1"] = probs[:, 1]
    out_path = output_dir / f"{target}_week{week}_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"[{target}] {len(out)} games -> {out_path.name}")
    return out

def main(targets, season, week, start_season, artifacts_dir, output_dir, margin_std, total_std, infer_week):
    artifacts_dir_p = Path(artifacts_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    schedule = get_current_week()
    if "season" not in schedule.columns:
        schedule["season"] = season

    print("=== DEBUG: Schedule DataFrame passed to predictions ===")
    print(schedule)
    print(f"Number of games in schedule: {len(schedule)}")

    if week is None:
        if infer_week:
            week = int(schedule["week"].max())
        else:
            raise ValueError("Week is None and infer_week disabled.")
    schedule_wk = schedule[schedule["week"] == week].copy()
    print(f"Number of games in week {week}: {len(schedule_wk)}")
    if schedule_wk.empty:
        raise ValueError(f"No schedule rows for week {week}")

    results = []
    for tgt in targets:
        try:
            res = predict_target(
                target=tgt,
                season=season,
                week=week,
                start_season=start_season,
                schedule_wk=schedule_wk,
                artifacts_dir=artifacts_dir_p,
                output_dir=output_dir_p,
                margin_std=margin_std,
                total_std=total_std
            )
            results.append(res)
        except Exception as e:
            print(f"[WARN] {tgt} failed: {e}", file=sys.stderr)

    if results:
        combined = pd.concat(results, ignore_index=True)
        combo_path = output_dir_p / f"all_targets_week{week}_predictions.csv"
        combined.to_csv(combo_path, index=False)
        print(f"Combined predictions -> {combo_path}")
    else:
        print("No predictions produced.")

if __name__ == "__main__":
    # Hardcoded for manual update each week
    season = 2025
    week = 3
    start_season = 2024  # must be <= season-1
    artifacts_dir = "artifacts"
    output_dir = "artifacts/predictions"
    margin_std = 13.5
    total_std = 13.5
    infer_week = False

    main(
        targets=ALL_TARGETS,
        season=season,
        week=week,
        start_season=start_season,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        margin_std=margin_std,
        total_std=total_std,
        infer_week=infer_week
    )