"""
Train player prop models for all position-prop combinations.

Positions:
- QB: pass_yds, pass_td_probability
- RB: rush_yds, rec_yds, receptions, any_time touchdowns
- PASS_CATCHER (WR/TE): rec_yds, rec_td_probability, receptions

Outputs:
- One pickled model artifact per position-prop-target combination
  Named: <position>_<prop>_<target>_xgb_<timestamp>.pkl
"""

from pathlib import Path
from datetime import datetime
from src.models.player_prop_model import NFLPlayerPropModel
from src.config.player_prop_config import PropType, PositionType, TargetType


# Define all model configurations to train
MODEL_CONFIGS = [
    # QB Models
    {"position": "QB", "prop_type": "pass_yds", "target_type": "yards"},
    {"position": "QB", "prop_type": "pass_yds", "target_type": "td_probability"},
    
    # RB Models
    {"position": "RB", "prop_type": "rush_yds", "target_type": "yards"},
    {"position": "RB", "prop_type": "rec_yds", "target_type": "yards"},
    {"position": "RB", "prop_type": "receptions", "target_type": "receptions"},
    {"position": "RB", "prop_type": "any_td", "target_type": "any_td_probability"},
    
    # WR/TE Models (combined as PASS_CATCHER)
    {"position": "PASS_CATCHER", "prop_type": "rec_yds", "target_type": "yards"},
    {"position": "PASS_CATCHER", "prop_type": "rec_yds", "target_type": "td_probability"},
    {"position": "PASS_CATCHER", "prop_type": "receptions", "target_type": "receptions"},
]


def train_model(position: PositionType, 
                prop_type: PropType, 
                target_type: TargetType,
                start_season: int = 2018,
                use_xgb: bool = True):
    """
    Train a single player prop model and persist artifact.
    
    Args:
        position: Player position
        prop_type: Type of prop to predict
        target_type: Yards, TD probability, or receptions
        start_season: Earliest season to include
        use_xgb: Use XGBoost (True) or Random Forest (False)
    """
    print(f"\n{'='*80}")
    print(f"Training: {position} | {prop_type} | {target_type}")
    print(f"{'='*80}\n")
    
    # Initialize model
    model = NFLPlayerPropModel(
        position=position,
        prop_type=prop_type,
        target_type=target_type
    )
    
    # Load data
    model.load_player_games(start_season=start_season)
    
    # Build features
    model.build_feature_matrices(
        include_opponent_defense=True,
        include_matchup_history=True
    )
    
    # Build dataset
    model.build_dataset()
    
    # Train model
    if use_xgb:
        metrics = model.fit_xgb()
        model_name = "xgb"
    else:
        metrics = model.fit_random_forest()
        model_name = "rf"
    
    # Persist artifact with timestamp
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path("artifacts/player_props")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = artifacts_dir / f"{position}_{prop_type}_{target_type}_{model_name}_{ts}.pkl"
    model.save(out_path.as_posix())
    
    print(f"\n[DONE] Metrics: {metrics}")
    print(f"[SAVED] {out_path}\n")
    
    return metrics


def main(start_season: int = 2018, use_xgb: bool = True):
    """Train all configured models."""
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Training {len(MODEL_CONFIGS)} Player Prop Models")
    print(f"Start Season: {start_season}")
    print(f"Model Type: {'XGBoost' if use_xgb else 'Random Forest'}")
    print(f"{'='*80}")
    
    for config in MODEL_CONFIGS:
        key = f"{config['position']}_{config['prop_type']}_{config['target_type']}"
        try:
            metrics = train_model(
                position=config['position'],
                prop_type=config['prop_type'],
                target_type=config['target_type'],
                start_season=start_season,
                use_xgb=use_xgb
            )
            results[key] = metrics
        except Exception as e:
            print(f"[ERROR] Failed to train {key}: {e}")
            results[key] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for key, metrics in results.items():
        if "error" in metrics:
            print(f"{key}: FAILED - {metrics['error']}")
        else:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"{key}: {metric_str}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Train with XGBoost by default
    main(start_season=2018, use_xgb=True)