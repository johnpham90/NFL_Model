from src.config.config import TeamFeatureConfigV2
from src.models.nfl_model import NFLModelV2


def test_feature_list_includes_requested_fields():
    config = TeamFeatureConfigV2()

    assert "time_of_possession_seconds" in config.all_canonical_features()
    assert "penalty_yards" in config.all_canonical_features()

    model = NFLModelV2()
    assert "time_of_possession_seconds" in model.stats
    assert "penalty_yards" in model.stats
    assert "penalties_yards" not in model.stats
