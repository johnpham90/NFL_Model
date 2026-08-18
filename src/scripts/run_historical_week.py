"""Train leakage-safe models and predict a historical NFL week."""

from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.models.nfl_model import NFLModelV2
from src.scripts.predictions import assemble_matchups
from src.utils.db_utils import execute_query


TARGETS = ("spread", "total_points", "binary_spread_label", "binary_ou_label")


def historical_schedule(season: int, week: int) -> pd.DataFrame:
    query = """
        SELECT season, week, date, hometeamid, awayteamid,
               spread, spreadfavoriteteam, over_under
        FROM stats.gamesummary
        WHERE season = :season AND week = :week
        ORDER BY date, gamesummaryid
    """
    schedule = execute_query(query, {"season": season, "week": week})
    if schedule is None or schedule.empty:
        raise ValueError(f"No schedule rows for season={season}, week={week}.")
    return schedule


def prior_games(model: NFLModelV2, season: int, week: int) -> None:
    mask = (model.games["season"] < season) | (
        (model.games["season"] == season) & (model.games["week"] < week)
    )
    model.games = model.games.loc[mask].copy()
    if model.games.empty:
        raise ValueError(f"No training games before season={season}, week={week}.")


def latest_team_history(model: NFLModelV2) -> pd.DataFrame:
    model.build_feature_matrices(
        exclude_current=True, include_defense=True, include_differentials=True
    )
    model.build_dataset()
    return (
        model._dataset.sort_values(["season", "week", "gamesummaryid"])
        .groupby("hometeamid")
        .tail(1)
        .set_index("hometeamid")
    )


def train_and_predict(season: int, week: int, start_season: int) -> None:
    schedule = historical_schedule(season, week)
    artifact_dir = Path("artifacts")
    output_dir = artifact_dir / "predictions"
    artifact_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    combined = []
    for target in TARGETS:
        base = NFLModelV2(target=target)
        base.load_games(start_season=start_season)
        prior_games(base, season, week)
        team_history = latest_team_history(base)

        trained = []
        for kind, fit_method in (
            ("rf", base.fit_random_forest),
            ("xgb", base.fit_xgb),
        ):
            metrics = fit_method()
            artifact = base.artifacts
            path = artifact_dir / f"{target}_{kind}_v2_{timestamp}.pkl"
            base.save(path.as_posix())

            design, meta = assemble_matchups(
                artifact, team_history, season, week, schedule
            )
            features = design[artifact.feature_columns]
            result = meta.copy()
            result["target"] = target
            result["model"] = kind
            result["prediction"] = artifact.model.predict(features)
            if hasattr(artifact.model, "predict_proba"):
                probabilities = artifact.model.predict_proba(features)
                if probabilities.shape[1] == 2:
                    result["prob_0"] = probabilities[:, 0]
                    result["prob_1"] = probabilities[:, 1]
            result.to_csv(
                output_dir / f"{target}_{kind}_week{week}_predictions.csv",
                index=False,
            )
            combined.append(result)
            trained.append(f"{kind}={metrics}")
        print(f"[DONE] {target}: " + "; ".join(trained))

    all_predictions = pd.concat(combined, ignore_index=True)
    out_path = output_dir / f"all_models_{season}_week{week}_predictions.csv"
    all_predictions.to_csv(out_path, index=False)
    print(f"[DONE] {len(all_predictions)} predictions -> {out_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--start-season", type=int, default=2010)
    args = parser.parse_args()
    train_and_predict(args.season, args.week, args.start_season)


if __name__ == "__main__":
    main()
