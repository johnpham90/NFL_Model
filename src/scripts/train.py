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
from src.models.xgboost_randomforrest_model_v2 import NFLModelV2

# List of targets to train. Adjust ordering or remove as needed.
TARGETS = [
    "spread",
    "total_points",
    "binary_spread_label",
    "binary_ou_label"
]

def train_target(target: str, start_season: int = 2021):
    """
    Train a single target model and persist artifact.

    Args:
        target: One of TARGETS.
        start_season: Earliest season (inclusive) to include.
    """
    model = NFLModelV2(target=target)
    model.load_games(start_season=start_season)
    # Build offensive + defensive + differential features
    model.build_feature_matrices(include_defense=True, include_differentials=True)
    model.build_dataset()
    # Random Forest baseline (swap to fit_xgb for XGBoost)
    metrics = model.fit_random_forest()
    # Persist artifact with timestamp
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    out_path = artifacts_dir / f"{target}_rf_v2_{ts}.pkl"
    model.save(out_path.as_posix())
    print(f"[DONE] {target}: {metrics} -> {out_path}")

def main():
    for tgt in TARGETS:
        train_target(tgt)

if __name__ == "__main__":
    main()