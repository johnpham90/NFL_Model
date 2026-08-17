"""DEPRECATED: This module is deprecated. Use config_v2.py instead.

Migration:
    from src.config.config_v2 import model_config_v2, team_feature_configs_v2
"""
import warnings
warnings.warn(
    "config.py is deprecated. "
    "Use config_v2.model_config_v2 and config_v2.team_feature_configs_v2 instead.",
    DeprecationWarning,
    stacklevel=2
)

# data:
#   raw_dir: data/raw
#   processed_dir: data/processed
#   external_dir: data/external

class model_config:
  train_test_split=.2
  start_season=2021
    
class team_feature_configs:
  parse_data_colums={"cmp_att_yd_td_int": ['completions', "pass attempts", "passing yards", "passing tds", "interceptions"],
                     "fourth_down_conv": ["fourth down conversions", "fourth down attempts"],
                     "rush_yds_tds": ["rush attempts", "rush yards", "rush tds"],
                     "fumbles_lost": ["fumbles", "fumbles lost"],
                     "sacked_yards": ["sacks", "sacked yards"],
                     "third_down_conv": ["third down success", "third down attempts"],
                     "penalties_yards": ["penalties", "penalty yards"]}
  efficiency_stats={"completion efficiency": ['completions', "pass attempts"], #first item in the list should be numerator, second should be denominator
                   "average completion": ["passing yards", "completions"],
                   "fourth down efficiency": ["fourth down conversions", "fourth down attempts"],
                   "rushing efficiency": ["rush yards", "rush attempts"],
                   "third down efficiency": ["third down success", "third down attempts"]                   
                   }
  team_features=["turnovers","total_yards","pass attempts","passing yards","third down attempts","fourth down conversions","fourth down attempts","first_downs","passing tds","interceptions","rush attempts","rush yards","rush tds","fumbles","fumbles lost","sacks","sacked yards","penalties","penalty yards","completion efficiency","average completion","fourth down efficiency","rushing efficiency","third down efficiency","team points"]

  