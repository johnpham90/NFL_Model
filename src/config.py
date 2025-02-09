# data:
#   raw_dir: data/raw
#   processed_dir: data/processed
#   external_dir: data/external

class model_config:
  train_test_split=.2
  start_season=2021
    
class team_feature_configs:
  parse_data_colums={"cmp_att_yd_td_int": ['completions', "pass attempts", "passing yards", "passing tds", "interceptions"],
                     "fourth_down_conv": ["fourth down converstions", "fourth down attempts"],
                     "rush_yds_tds": ["rush attempts", "rush yards", "rush tds"],
                     "fumbles_lost": ["fumbles", "fumbles lost"],
                     "sacked_yards": ["sacks", "sacked yards"],
                     "third_down_conv": ["third down success", "third down attempts"],
                     "penalties_yards": ["penalties", "penalty yards"]}
  efficiency_stats={"completion efficiency": ['completions', "pass attempts"], #first item in the list should be numerator, second should be denominator
                   "average completion": ["passing yards", "completions"],
                   "fourth down efficiency": ["fourth down converstions", "fourth down attempts"],
                   "rushing efficiency": ["rush yards", "rush attempts"],
                   "third down efficiency": ["third down success", "third down attempts"]                   
                   }
  team_features=["turnovers","total_yards","first_downs","passing tds","interceptions","rush attempts","rush yards","rush tds","fumbles","fumbles lost","sacks","sacked yards","penalties","penalty yards","completion efficiency","average completion","fourth down efficiency","rushing efficiency","third down efficiency","team points"]

  