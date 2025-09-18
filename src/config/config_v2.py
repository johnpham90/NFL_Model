"""v2 configuration mirroring legacy structure with added canonical helpers."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfigV2:
    start_season: int = 2021
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class TeamFeatureConfigV2:
    # Legacy names (with spaces / typos) preserved
    parse_data_colums: Dict[str, List[str]] = field(default_factory=lambda: {
        "cmp_att_yd_td_int": ['completions', "pass attempts", "passing yards", "passing tds", "interceptions"],
        "fourth_down_conv": ["fourth down converstions", "fourth down attempts"],
        "rush_yds_tds": ["rush attempts", "rush yards", "rush tds"],
        "fumbles_lost": ["fumbles", "fumbles lost"],
        "sacked_yards": ["sacks", "sacked yards"],
        "third_down_conv": ["third down success", "third down attempts"],
        "penalties_yards": ["penalties", "penalty yards"],
    })
    efficiency_stats: Dict[str, List[str]] = field(default_factory=lambda: {
        "completion efficiency": ['completions', "pass attempts"],
        "average completion": ["passing yards", "completions"],
        "fourth down efficiency": ["fourth down converstions", "fourth down attempts"],
        "rushing efficiency": ["rush yards", "rush attempts"],
        "third down efficiency": ["third down success", "third down attempts"],
    })
    team_features: List[str] = field(default_factory=lambda: [
        "turnovers","total_yards","pass attempts","passing yards","third down attempts",
        "fourth down converstions","fourth down attempts","first_downs","passing tds",
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
        "fourth down converstions": "fourth_down_conversions",
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


@dataclass
class PlayerFeatureConfigV2:
    # Position-specific stat groups for rolling averages
    position_stat_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "QB": [
            # Passing stats (primary for QB)
            "pass_yds", "pass_td", "pass_int", "pass_cmp", "pass_att", "pass_rating",
            "pass_long", "pass_sacked", "pass_sacked_yds",
            # Rushing stats (QB mobility)
            "rush_yds", "rush_td", "rush_att", "rush_long",
            # Turnovers
            "fumbles", "fumbles_lost"
        ],
        "RB": [
            # Rushing stats (primary for RB)
            "rush_yds", "rush_td", "rush_att", "rush_long",
            # Receiving stats (pass-catching backs)
            "rec_yds", "rec_td", "rec", "targets", "rec_long",
            # Turnovers
            "fumbles", "fumbles_lost"
        ],
        "WR": [
            # Receiving stats (primary for WR)
            "rec_yds", "rec_td", "rec", "targets", "rec_long",
            # Occasional rushing (end-arounds, etc.)
            "rush_yds", "rush_td", "rush_att", "rush_long",
            # Turnovers
            "fumbles", "fumbles_lost"
        ],
        "TE": [
            # Receiving stats (primary for TE)
            "rec_yds", "rec_td", "rec", "targets", "rec_long",
            # Turnovers
            "fumbles", "fumbles_lost"
        ]
    })
    
    # Player rankings we want to track per team per game
    position_rankings: List[str] = field(default_factory=lambda: [
        "QB1",    # Top QB by pass attempts
        "RB1",    # Top RB by rush attempts  
        "TE1",    # Top TE by targets
        "WR1",    # Top WR by targets
        "WR2",    # 2nd WR by targets
        "WR3"     # 3rd WR by targets
    ])
    
    # Rolling window sizes (matching team features)
    rolling_windows: List[int] = field(default_factory=lambda: [2, 5, 10])
    
    # Minimum games required for rolling averages
    min_games_required: int = 1
    
    # Position ranking criteria (stat used to determine top player)
    ranking_criteria: Dict[str, str] = field(default_factory=lambda: {
        "QB": "pass_att",     # Top QB by pass attempts
        "RB": "rush_att",     # Top RB by rush attempts
        "WR": "targets",      # Top WRs by targets
        "TE": "targets"       # Top TE by targets
    })
    
    def get_position_stats(self, position: str) -> List[str]:
        """Get relevant stats for a position."""
        return self.position_stat_groups.get(position, [])
    
    def get_ranking_stat(self, position: str) -> str:
        """Get the stat used to rank players at this position."""
        return self.ranking_criteria.get(position, "targets")


model_config_v2 = ModelConfigV2()
team_feature_configs_v2 = TeamFeatureConfigV2()
player_feature_configs_v2 = PlayerFeatureConfigV2()

__all__ = [
    "ModelConfigV2",
    "TeamFeatureConfigV2", 
    "PlayerFeatureConfigV2",
    "model_config_v2",
    "team_feature_configs_v2",
    "player_feature_configs_v2",
]