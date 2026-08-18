"""v2 configuration mirroring legacy structure with added canonical helpers."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfigV2:
    start_season: int = 2010
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class TeamFeatureConfigV2:
    # Legacy names (with spaces / typos) preserved
    parse_data_colums: Dict[str, List[str]] = field(default_factory=lambda: {
        "cmp_att_yd_td_int": ['completions', "pass attempts", "passing yards", "passing tds", "interceptions"],
        "fourth_down_conv": ["fourth down conversions", "fourth down attempts"],
        "rush_yds_tds": ["rush attempts", "rush yards", "rush tds"],
        "fumbles_lost": ["fumbles", "fumbles lost"],
        "sacked_yards": ["sacks", "sacked yards"],
        "third_down_conv": ["third down success", "third down attempts"],
        "penalties_yards": ["penalties", "penalty yards"],
    })
    efficiency_stats: Dict[str, List[str]] = field(default_factory=lambda: {
        "completion efficiency": ['completions', "pass attempts"],
        "average completion": ["passing yards", "completions"],
        "fourth down efficiency": ["fourth down conversions", "fourth down attempts"],
        "rushing efficiency": ["rush yards", "rush attempts"],
        "third down efficiency": ["third down success", "third down attempts"],
    })
    team_features: List[str] = field(default_factory=lambda: [
        "turnovers","total_yards","pass attempts","passing yards","third down attempts",
        "fourth down conversions","fourth down attempts","first_downs","passing tds",
        "interceptions","rush attempts","rush yards","rush tds","fumbles","fumbles lost",
        "sacks","sacked yards","penalties","penalty yards","team points",
        "completion efficiency","average completion","fourth down efficiency",
        "rushing efficiency","third down efficiency",
    ])
    # Canonical mapping and lists
    legacy_to_canonical: Dict[str, str] = field(default_factory=lambda: {
        "pass attempts": "pass_attempts",
        "passing yards": "passing_yards",
        "third down attempts": "third_down_attempts",
        "fourth down conversions": "fourth_down_conversions",
        "fourth down attempts": "fourth_down_attempts",
        "passing tds": "passing_tds",
        "rush attempts": "rush_attempts",
        "rush yards": "rush_yards",
        "rush tds": "rush_tds",
        "fumbles lost": "fumbles_lost",
        "sacked yards": "sacked_yards",
        "penalty yards": "penalty_yards",
        "team points": "team_points",
        "completion efficiency": "completion_efficiency",
        "average completion": "yards_per_completion",
        "fourth down efficiency": "fourth_down_efficiency",
        "rushing efficiency": "rushing_efficiency",
        "third down efficiency": "third_down_efficiency",
        "third down success": "third_down_success",
    })
    canonical_base_team_features: List[str] = field(default_factory=lambda: [
        "turnovers","total_yards","pass_attempts","passing_yards","third_down_attempts",
        "fourth_down_conversions","fourth_down_attempts","first_downs","passing_tds",
        "interceptions","rush_attempts","rush_yards","rush_tds","fumbles","fumbles_lost",
        "sacks","sacked_yards","penalties","penalty_yards","team_points",
    ])
    canonical_efficiency_specs: Dict[str, List[str]] = field(default_factory=lambda: {
        "completion_efficiency": ["completions","pass_attempts"],
        "yards_per_completion": ["passing_yards","completions"],
        "fourth_down_efficiency": ["fourth_down_conversions","fourth_down_attempts"],
        "rushing_efficiency": ["rush_yards","rush_attempts"],
        "third_down_efficiency": ["third_down_success","third_down_attempts"],
    })
    # Rolling window sizes for parity with legacy feature engineering
    rolling_windows: List[int] = field(default_factory=lambda: [2, 5, 10])

    def all_features(self) -> List[str]:  # legacy style
        return list(self.team_features)

    def all_canonical_features(self) -> List[str]:
        return list(self.canonical_base_team_features) + list(self.canonical_efficiency_specs.keys())

    def canonicalize(self, name: str) -> str:
        return self.legacy_to_canonical.get(name, name)


model_config = ModelConfigV2()
team_feature_configs = TeamFeatureConfigV2()

# Aliases for backward compatibility
model_config_v2 = model_config
team_feature_configs_v2 = team_feature_configs

__all__ = [
    "ModelConfigV2",
    "TeamFeatureConfigV2",
    "model_config",
    "team_feature_configs",
    "model_config_v2",  # alias
    "team_feature_configs_v2",  # alias
]

TARGETS = [
    "spread",
    "total_points",
    "binary_spread_label",
    "binary_ou_label"
]

models={"binary_spread_label":"classification", 
        "binary_ou_label":"classification",
        "spread":"regression",
        "total_points":"regression"}
