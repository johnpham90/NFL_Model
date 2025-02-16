
#1 Redundant Code Patterns
# the  code heavily repeats the same pattern for different statistics (net_pass_yards_stat, turnovers_stat, total_yards_stat)
# These could be consolidated into a single generic method like:


def calculate_team_stat(self, stat_name: str, data_column: str):
    df_offense = pd.DataFrame(index=self.master_index, columns=self.teams)
    df_defense = pd.DataFrame(index=self.master_index, columns=self.teams)
    
    for perspective in ['offense', 'defense']:
        id_column = 'teamid' if perspective == 'offense' else 'defenseid'
        df = df_offense if perspective == 'offense' else df_defense
        
        for team in self.teams:
            for season in self.seasons:
                season_idx = np.where((self.data[id_column] == team) & 
                                    (self.data['season'] == season))[0]
                


#2 Initialization is Doing Too Much
#The __init__ method is handling data loading AND multiple stat calculations
# this violates the Single Responsibility Principle
# Better structure would be:


 def __init__(self):
    self.data = self._load_data()
    self._initialize_base_structures()

def _load_data(self):
    query = f"""SELECT * FROM stats.teamstats 
               WHERE season > {config.model_config.start_season}"""
    return execute_query(query)

def _initialize_base_structures(self):
    self.teams = list(np.unique(self.data['teamid']))
    self.seasons = list(np.unique(self.data['season']))
    self.build_defense_id()
    self.build_master_index()

def calculate_all_stats(self):
    # Move stat calculations here
    self.net_pass_yards_stat()
    self.turnovers_stat()
    # etc...

## 3 Fix Spelling error

## 4 Documentations
