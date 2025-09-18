import pandas as pd
import numpy as np
import glob
import pickle
from pathlib import Path
from math import sqrt
from scipy.stats import norm


def find_latest_artifact(pattern: str, artifacts_dir: str = "artifacts") -> Path:
    files = sorted(glob.glob(str(Path(artifacts_dir) / f"*{pattern}*.pkl")), reverse=True)
    return Path(files[0]) if files else None


def load_artifact(path: Path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception:
        return None


def infer_sigma_from_artifact(artifact) -> float:
    # Try common attribute names, fall back to sensible default
    if artifact is None:
        return 12.5
    # common locations: artifact.rmse, artifact.residual_std, artifact.metrics
    if hasattr(artifact, "rmse") and artifact.rmse:
        return float(artifact.rmse)
    if hasattr(artifact, "residual_std") and artifact.residual_std:
        return float(artifact.residual_std)
    # some artifacts store metrics dict
    metrics = None
    if hasattr(artifact, "metrics") and isinstance(artifact.metrics, dict):
        metrics = artifact.metrics
    elif hasattr(artifact, "model") and hasattr(artifact, "model"):
        # some model wrappers keep metrics on the wrapper
        metrics = getattr(artifact, "metrics", None)
    if metrics:
        # prefer RMSE if present, else MAE -> approximate RMSE
        if "rmse" in metrics:
            return float(metrics["rmse"])
        if "mae" in metrics:
            # convert MAE -> RMSE rough approximation
            try:
                mae = float(metrics["mae"])
                return max(8.0, mae * 1.25)
            except Exception:
                pass
    # fallback default
    return 12.5


def annotate_binaries(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> pd.DataFrame:
    df = df.copy()
    if "target" not in df.columns:
        return df
    binary_rows = df[df["target"].str.startswith("binary_")]
    if binary_rows.empty:
        return df

    for target in binary_rows["target"].unique():
        art_path = find_latest_artifact(target, artifacts_dir=artifacts_dir)
        if not art_path:
            print(f"No artifact found for {target} to infer classes")
            continue
        art = load_artifact(art_path)
        classes = tuple(getattr(art.model, "classes_", (0, 1))) if art else (0, 1)
        print(f"Artifact for {target}: {art_path.name} -> model.classes_ = {classes}")

        # meaning mapping based on actual training code
        if target == "binary_spread_label":
            meaning = {classes[0]: "FAVORITE did NOT cover", classes[1]: "FAVORITE covered"}
        elif target == "binary_ou_label":
            meaning = {classes[0]: "UNDER", classes[1]: "OVER"}
        else:
            meaning = {classes[0]: f"class_{classes[0]}", classes[1]: f"class_{classes[1]}"}

        mask = df["target"] == target
        # prefer explicit prediction column
        if "prediction" in df.columns and df.loc[mask, "prediction"].notna().any():
            df.loc[mask, "binary_value"] = df.loc[mask, "prediction"].astype(int)
        else:
            if "prob_0" in df.columns and "prob_1" in df.columns:
                df.loc[mask, "binary_value"] = (df.loc[mask, "prob_1"].astype(float) >= df.loc[mask, "prob_0"].astype(float)).astype(int)
            else:
                df.loc[mask, "binary_value"] = None

        df.loc[mask, "model_class_label"] = df.loc[mask, "binary_value"].map({0: classes[0], 1: classes[1]})
        df.loc[mask, "binary_meaning"] = df.loc[mask, "model_class_label"].map(meaning)

    return df


def find_latest_all_targets(predictions_dir: str = "artifacts/predictions") -> Path:
    pattern = str(Path(predictions_dir) / "all_targets_week*_predictions.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Sort by week number extracted from filename
    def extract_week(filepath):
        import re
        match = re.search(r'week(\d+)', Path(filepath).name)
        return int(match.group(1)) if match else 0
    
    latest_file = max(files, key=extract_week)
    return Path(latest_file)


def analyze_spreads(df: pd.DataFrame, sigma: float):
    """Analyze spread predictions"""
    spread_df = df[df["target"] == "spread"].copy()
    if spread_df.empty:
        print("No spread predictions found")
        return
    
    print("\n" + "="*60)
    print("SPREAD ANALYSIS")
    print("="*60)
    
    # model edge relative to market (higher -> model favors home more than market)
    spread_df["model_minus_market"] = spread_df["prediction"].astype(float) - spread_df["market_line_home"].astype(float)

    print("Edge stats (model - market):")
    print(spread_df["model_minus_market"].describe().round(3))

    print("\nSpread prediction distribution:")
    print(spread_df["prediction"].describe().round(3))

    print("\nModel expects home to cover in {:.1%} of games".format((spread_df["model_minus_market"] > 0).mean()))

    # calibrated probabilities
    spread_df["prob_home_win_calibrated"] = 1.0 - norm.cdf(0.0, loc=spread_df["prediction"].astype(float), scale=sigma)
    spread_df["prob_home_covers_calibrated"] = 1.0 - norm.cdf(spread_df["market_line_home"].astype(float), loc=spread_df["prediction"].astype(float), scale=sigma)

    # show extremes for inspection
    print("\nTop positive spread edges (model >> market):")
    cols = ["hometeamid", "awayteamid", "market_line_home", "prediction", "model_minus_market", "prob_home_covers_calibrated"]
    print(spread_df.sort_values("model_minus_market", ascending=False)[cols].head(10).to_string(index=False))

    print("\nTop negative spread edges (model << market):")
    print(spread_df.sort_values("model_minus_market")[cols].head(10).to_string(index=False))

    # suggested betting candidates
    EDGE_THRESHOLD = 3.0
    PROB_THRESHOLD = 0.60
    candidates = spread_df[(spread_df["prob_home_covers_calibrated"] >= PROB_THRESHOLD) & (spread_df["model_minus_market"] >= EDGE_THRESHOLD)]
    print(f"\nSuggested SPREAD betting candidates (cover_prob >= {PROB_THRESHOLD} and edge >= {EDGE_THRESHOLD}):")
    if candidates.empty:
        print("  None")
    else:
        print(candidates[cols].to_string(index=False))


def analyze_totals(df: pd.DataFrame, sigma: float):
    """Analyze total points (O/U) predictions"""
    totals_df = df[df["target"] == "total_points"].copy()
    if totals_df.empty:
        print("No total points predictions found")
        return
    
    print("\n" + "="*60)
    print("OVER/UNDER ANALYSIS")
    print("="*60)
    
    # model edge relative to market (higher -> model expects more points than market)
    totals_df["model_minus_market"] = totals_df["prediction"].astype(float) - totals_df["market_total"].astype(float)

    print("O/U Edge stats (model - market total):")
    print(totals_df["model_minus_market"].describe().round(3))

    print("\nTotal points prediction distribution:")
    print(totals_df["prediction"].describe().round(3))

    print("\nModel expects OVER in {:.1%} of games".format((totals_df["model_minus_market"] > 0).mean()))

    # calibrated probabilities for over/under
    totals_df["prob_over_calibrated"] = 1.0 - norm.cdf(totals_df["market_total"].astype(float), loc=totals_df["prediction"].astype(float), scale=sigma)
    totals_df["prob_under_calibrated"] = 1.0 - totals_df["prob_over_calibrated"]

    # show extremes for inspection
    print("\nTop positive O/U edges (model expects higher scoring):")
    cols = ["hometeamid", "awayteamid", "market_total", "prediction", "model_minus_market", "prob_over_calibrated"]
    print(totals_df.sort_values("model_minus_market", ascending=False)[cols].head(10).to_string(index=False))

    print("\nTop negative O/U edges (model expects lower scoring):")
    cols_under = ["hometeamid", "awayteamid", "market_total", "prediction", "model_minus_market", "prob_under_calibrated"]
    print(totals_df.sort_values("model_minus_market")[cols_under].head(10).to_string(index=False))

    # suggested betting candidates
    EDGE_THRESHOLD = 3.0
    PROB_THRESHOLD = 0.60
    
    over_candidates = totals_df[(totals_df["prob_over_calibrated"] >= PROB_THRESHOLD) & (totals_df["model_minus_market"] >= EDGE_THRESHOLD)]
    under_candidates = totals_df[(totals_df["prob_under_calibrated"] >= PROB_THRESHOLD) & (totals_df["model_minus_market"] <= -EDGE_THRESHOLD)]
    
    print(f"\nSuggested OVER betting candidates (over_prob >= {PROB_THRESHOLD} and edge >= {EDGE_THRESHOLD}):")
    if over_candidates.empty:
        print("  None")
    else:
        print(over_candidates[cols].to_string(index=False))
    
    print(f"\nSuggested UNDER betting candidates (under_prob >= {PROB_THRESHOLD} and edge <= -{EDGE_THRESHOLD}):")
    if under_candidates.empty:
        print("  None")
    else:
        print(under_candidates[cols_under].to_string(index=False))


def analyze_binaries(df: pd.DataFrame):
    """Analyze binary predictions"""
    binary_df = df[df["target"].str.startswith("binary_")].copy()
    if binary_df.empty:
        print("No binary predictions found")
        return
    
    print("\n" + "="*60)
    print("BINARY PREDICTIONS ANALYSIS")
    print("="*60)
    
    for target in binary_df["target"].unique():
        target_df = binary_df[binary_df["target"] == target].copy()
        print(f"\n{target.upper()}:")
        
        if "prob_0" in target_df.columns and "prob_1" in target_df.columns:
            target_df["confidence"] = abs(target_df["prob_1"].astype(float) - target_df["prob_0"].astype(float))
            target_df["predicted_class"] = (target_df["prob_1"].astype(float) > target_df["prob_0"].astype(float)).astype(int)
            
            print(f"  Class distribution: {target_df['predicted_class'].value_counts().to_dict()}")
            print(f"  Average confidence: {target_df['confidence'].mean():.3f}")
            
            # High confidence predictions
            high_conf = target_df[target_df["confidence"] >= 0.3].sort_values("confidence", ascending=False)
            if not high_conf.empty:
                print(f"\n  High confidence predictions (confidence >= 0.3):")
                cols = ["hometeamid", "awayteamid", "predicted_class", "prob_0", "prob_1", "confidence"]
                print(high_conf[cols].head(10).to_string(index=False))


def main():
    pred_csv = find_latest_all_targets()
    if not pred_csv or not pred_csv.exists():
        print("No all_targets prediction CSV found in artifacts/predictions/")
        return
    
    print(f"Using prediction file: {pred_csv}")
    df = pd.read_csv(pred_csv)
    
    print(f"Loaded {len(df)} predictions")
    print(f"Target types: {df['target'].unique()}")
    print(f"Games: {df[['hometeamid', 'awayteamid']].drop_duplicates().shape[0]}")

    # Get sigma from spread artifact for calibrated probabilities
    spread_art = find_latest_artifact("spread_rf_v2") or find_latest_artifact("spread_rf")
    artifact = load_artifact(spread_art) if spread_art else None
    sigma = infer_sigma_from_artifact(artifact)
    print(f"\nUsing sigma (residual std) = {sigma:.2f} for calibrated probabilities")

    # Analyze each target type
    analyze_spreads(df, sigma)
    analyze_totals(df, sigma)  
    analyze_binaries(df)

    # annotate binary rows using artifacts
    out_df = annotate_binaries(df)

    out_path = pred_csv.with_name(pred_csv.stem + "_annotated.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote annotated predictions to: {out_path}")


if __name__ == "__main__":
    main()