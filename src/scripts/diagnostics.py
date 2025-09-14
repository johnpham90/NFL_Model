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

        # default meaning mapping (adjust if your training used different semantics)
        if target == "binary_spread_label":
            meaning = {classes[0]: "fav DID NOT cover / home DID NOT win", classes[1]: "fav COVERED / home WON"}
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


def main():
    pred_csv = Path("artifacts/predictions/spread_week2_predictions.csv")
    if not pred_csv.exists():
        print("Prediction CSV not found:", pred_csv)
        return
    df = pd.read_csv(pred_csv)

    # model edge relative to market (higher -> model favors home more than market)
    df["model_minus_market"] = df["prediction"].astype(float) - df["market_line_home"].astype(float)

    print("Edge stats (model - market):")
    print(df["model_minus_market"].describe().round(3))

    print("\nPrediction distribution:")
    print(df["prediction"].describe().round(3))

    print("\nModel expects cover in {:.1%} of games".format((df["model_minus_market"] > 0).mean()))

    # pick a regression artifact (spread) to infer sigma (RMSE) if available
    spread_art = find_latest_artifact("spread_rf_v2") or find_latest_artifact("spread_rf")
    artifact = load_artifact(spread_art) if spread_art else None
    sigma = infer_sigma_from_artifact(artifact)
    print(f"Using sigma (residual std) = {sigma:.2f} for calibrated probabilities")

    # calibrated probabilities
    df["prob_home_win_calibrated"] = 1.0 - norm.cdf(0.0, loc=df["prediction"].astype(float), scale=sigma)
    df["prob_home_covers_calibrated"] = 1.0 - norm.cdf(df["market_line_home"].astype(float), loc=df["prediction"].astype(float), scale=sigma)

    # show extremes for inspection
    print("\nTop positive edges (model >> market):")
    print(df.sort_values("model_minus_market", ascending=False)[["hometeamid", "awayteamid", "market_line_home", "prediction", "model_minus_market", "cover_prob"]].head(10).to_string(index=False))

    print("\nTop negative edges (model << market):")
    print(df.sort_values("model_minus_market")[ ["hometeamid", "awayteamid", "market_line_home", "prediction", "model_minus_market", "cover_prob"] ].head(10).to_string(index=False))

    # suggested betting candidates
    EDGE_THRESHOLD = 3.0
    PROB_THRESHOLD = 0.60
    candidates = df[(df["prob_home_covers_calibrated"] >= PROB_THRESHOLD) & (df["model_minus_market"] >= EDGE_THRESHOLD)]
    print(f"\nSuggested betting candidates (cover_prob >= {PROB_THRESHOLD} and edge >= {EDGE_THRESHOLD}):")
    if candidates.empty:
        print("  None")
    else:
        print(candidates[["hometeamid", "awayteamid", "market_line_home", "prediction", "model_minus_market", "prob_home_covers_calibrated"]].to_string(index=False))

    # annotate binary rows using artifacts
    out_df = annotate_binaries(df)

    out_path = pred_csv.with_name(pred_csv.stem + "_annotated.csv")
    out_df.to_csv(out_path, index=False)
    print("Wrote annotated predictions to:", out_path)


if __name__ == "__main__":
    main()
