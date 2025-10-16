"""Player prop model configuration mirroring team-level config structure."""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

PropType = Literal["pass_yds", "rush_yds", "rec_yds", "receptions", "any_td"]
PositionType = Literal["QB", "RB", "WR", "TE", "PASS_CATCHER"]
TargetType = Literal["yards", "td_probability", "receptions", "any_td_probability"]


# ========== CENTRALIZED MAPPINGS ==========
# These mappings are used across multiple feature files to ensure consistency

# Prop type to target column mappings
PROP_TYPE_TO_TARGET_COL: Dict[str, str] = {
    "pass_yds": "pass_yds",
    "rush_yds": "rush_yds",
    "rec_yds": "rec_yds",
    "receptions": "rec",
    "any_td": "has_any_td"
}

# Prop type to opponent defense target mappings
PROP_TYPE_TO_DEFENSE_TARGET: Dict[str, Tuple[str, str]] = {
    'pass_yds': ('pass_yds', 'opp_pass_yds_allowed_hist'),
    'rush_yds': ('rush_yds', 'opp_rush_yds_allowed_hist'),
    'rec_yds': ('rec_yds', 'opp_rec_yds_allowed_hist'),
    'receptions': ('rec', 'opp_rec_allowed_hist')
}

# Model target type to column mappings
TARGET_TYPE_TO_COL: Dict[Tuple[str, str], str] = {
    ("yards", "pass_yds"): "pass_yds",
    ("yards", "rush_yds"): "rush_yds",
    ("yards", "rec_yds"): "rec_yds",
    ("td_probability", "pass_yds"): "has_pass_td",
    ("td_probability", "rush_yds"): "has_rush_td",
    ("td_probability", "rec_yds"): "has_rec_td",
    ("any_td_probability", "any_td"): "has_any_td",
    ("receptions", "receptions"): "rec"
}

# Position to snap count position mappings
POSITION_TO_SNAP_POS: Dict[str, List[str]] = {
    'QB': ['QB'],
    'RB': ['RB'],
    'PASS_CATCHER': ['RB', 'WR', 'TE'],
    'WR': ['WR'],
    'TE': ['TE']
}


@dataclass
class PlayerPropModelConfig:
    """Basic model configuration."""
    start_season: int = 2018
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class PlayerPropFeatureConfig:
    """Feature engineering configuration for player props."""
    
    # Rolling window sizes for historical averages
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    # ========== PASSING FEATURES (QB) ==========
    passing_volume_stats: List[str] = field(default_factory=lambda: [
        "pass_att", "pass_cmp"
    ])
    
    passing_yardage_stats: List[str] = field(default_factory=lambda: [
        "pass_yds", "pass_air_yds", "pass_yac"
    ])
    
    passing_efficiency_stats: List[str] = field(default_factory=lambda: [
        "pass_air_yds_per_att", "pass_air_yds_per_cmp",
        "pass_yac_per_cmp", "pass_tgt_yds_per_att",
        "pass_poor_throw_pct", "pass_drop_pct",
        "pass_first_down_pct"
    ])
    
    passing_pressure_stats: List[str] = field(default_factory=lambda: [
        "pass_pressured_pct", "pass_sacked", "pass_blitzed",
        "pass_hits", "pass_hurried"
    ])
    
    passing_td_stats: List[str] = field(default_factory=lambda: [
        "pass_td", "pass_int", "pass_rating"
    ])
    
    # QB rushing stats (as features for passing models)
    qb_rushing_stats: List[str] = field(default_factory=lambda: [
        "qb_rush_att", "qb_rush_yds", "qb_rush_td", 
        "qb_rush_yac", "qb_rush_broken_tackles"
    ])

    # ========== RUSHING FEATURES (RB/QB) ==========
    rushing_volume_stats: List[str] = field(default_factory=lambda: [
        "rush_att"
    ])
    
    rushing_yardage_stats: List[str] = field(default_factory=lambda: [
        "rush_yds", "rush_yac", "rush_yds_before_contact"
    ])
    
    rushing_efficiency_stats: List[str] = field(default_factory=lambda: [
        "rush_yac_per_rush", "rush_yds_bc_per_rush",
        "rush_broken_tackles_per_rush"
    ])
    
    rushing_td_stats: List[str] = field(default_factory=lambda: [
        "rush_td", "fumbles", "fumbles_lost"
    ])
    
    # ========== RECEIVING FEATURES (WR/TE/RB) ==========
    receiving_volume_stats: List[str] = field(default_factory=lambda: [
        "targets", "rec"
    ])
    
    receiving_yardage_stats: List[str] = field(default_factory=lambda: [
        "rec_yds", "rec_air_yds", "rec_yac"
    ])
    
    receiving_efficiency_stats: List[str] = field(default_factory=lambda: [
        "rec_adot", "rec_air_yds_per_rec", "rec_yac_per_rec",
        "rec_drop_pct", "rec_broken_tackles_per_rec",
        "catch_rate"
    ])
    
    receiving_td_stats: List[str] = field(default_factory=lambda: [
        "rec_td", "rec_pass_rating", "rec_target_int"
    ])
    
    # ========== EFFICIENCY CALCULATION SPECS ==========
    # Format: {efficiency_name: (numerator_col, denominator_col)}
    passing_efficiency_specs: Dict[str, tuple] = field(default_factory=lambda: {
        "completion_pct": ("pass_cmp", "pass_att"),
        "yds_per_att": ("pass_yds", "pass_att"),
        "yds_per_cmp": ("pass_yds", "pass_cmp"),
        "td_rate": ("pass_td", "pass_att"),
        "int_rate": ("pass_int", "pass_att"),
    })
    
    rushing_efficiency_specs: Dict[str, tuple] = field(default_factory=lambda: {
        "yds_per_carry": ("rush_yds", "rush_att"),
        "td_rate": ("rush_td", "rush_att"),
        "fumble_rate": ("fumbles", "rush_att"),
    })
    
    receiving_efficiency_specs: Dict[str, tuple] = field(default_factory=lambda: {
        "catch_rate": ("rec", "targets"),
        "yds_per_target": ("rec_yds", "targets"),
        "yds_per_rec": ("rec_yds", "rec"),
        "td_rate": ("rec_td", "targets"),
    })
    
    def get_all_passing_features(self) -> List[str]:
        """Return all passing stat names."""
        return (self.passing_volume_stats + 
                self.passing_yardage_stats + 
                self.passing_efficiency_stats + 
                self.passing_pressure_stats +
                self.passing_td_stats +
                self.qb_rushing_stats) 
    
    def get_all_rushing_features(self) -> List[str]:
        """Return all rushing stat names."""
        return (self.rushing_volume_stats + 
                self.rushing_yardage_stats + 
                self.rushing_efficiency_stats +
                self.rushing_td_stats)
    
    def get_all_receiving_features(self) -> List[str]:
        """Return all receiving stat names."""
        return (self.receiving_volume_stats + 
                self.receiving_yardage_stats + 
                self.receiving_efficiency_stats +
                self.receiving_td_stats)
    
    def get_features_for_prop(self, prop_type: PropType) -> List[str]:
        """Get relevant features based on prop type."""
        if prop_type == "pass_yds":
            return self.get_all_passing_features()
        elif prop_type == "rush_yds":
            return self.get_all_rushing_features()
        elif prop_type in ["rec_yds", "receptions"]:
            return self.get_all_receiving_features()
        elif prop_type == "any_td":
            # Combined RB model needs both rushing and receiving features
            return self.get_all_rushing_features() + self.get_all_receiving_features()
        else:
            raise ValueError(f"Unknown prop_type: {prop_type}")


# Global instances
player_prop_model_config = PlayerPropModelConfig()
player_prop_feature_config = PlayerPropFeatureConfig()

__all__ = [
    "PlayerPropModelConfig",
    "PlayerPropFeatureConfig",
    "player_prop_model_config",
    "player_prop_feature_config",
    "PropType",
    "PositionType",
    "TargetType",
    "PROP_TYPE_TO_TARGET_COL",
    "PROP_TYPE_TO_DEFENSE_TARGET",
    "TARGET_TYPE_TO_COL",
    "POSITION_TO_SNAP_POS",
]