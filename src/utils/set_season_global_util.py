class QueryParams:
    """
    A utility class to store and generate SQL query parameters with default values pre-set.
    To use utility:
    Import utilit to file: from src.utils.set_season_global_util import QueryParams
    Set QueryParams.get_where_clause() as a variable
    Use {varible} as part of SQL statement
    Output: WHERE season = 'input params'
    """
    # Default parameters and conditions set within the utility
    _params = {
        "season": ["2004-2020"],  # Handle mixed multiple seasons and ranges ie "2004,2005" "2004-2020"
        "week": None,
        "away_team_id": None,
        "home_team_id": None
    }

    _season_range = []

    @classmethod
    def _initialize_conditions(cls):
        cls._season_range = []
        season = cls._params["season"]

        # Handle different season types
        if isinstance(season, list):  # Handle list of seasons and ranges
            conditions = []
            for s in season:
                if ',' in s:  # Multiple seasons in a string
                    seasons = ", ".join(f"'{item.strip()}'" for item in s.split(','))
                    conditions.append(f"season IN ({seasons})")
                elif '-' in s:  # Season range
                    start, end = map(int, s.split('-'))
                    conditions.append(f"season BETWEEN {start} AND {end}")
                else:  # Single season
                    conditions.append(f"season = '{s}'")
            cls._season_range.append(f"({' OR '.join(conditions)})")
        elif isinstance(season, str) and ',' in season:  # Multiple seasons in a string
            seasons = ", ".join(f"'{s.strip()}'" for s in season.split(','))
            cls._season_range.append(f"season IN ({seasons})")
        elif isinstance(season, str) and '-' in season:  # Season range
            start, end = map(int, season.split('-'))
            cls._season_range.append(f"season BETWEEN {start} AND {end}")
        elif season:  # Single season
            cls._season_range.append(f"season = '{season}'")

        # Add additional parameters
        if cls._params["week"] is not None:
            cls._season_range.append(f"week = {cls._params['week']}")
        if cls._params["away_team_id"] is not None:
            cls._season_range.append(f"away_team_id = '{cls._params['away_team_id']}'")
        if cls._params["home_team_id"] is not None:
            cls._season_range.append(f"home_team_id = '{cls._params['home_team_id']}'")

    @classmethod
    def initialize(cls):
        """Initialize conditions based on pre-set parameters in the utility."""
        cls._initialize_conditions()

    @classmethod
    def get_where_clause(cls):
        """
        Generate a WHERE clause using the pre-set parameters defined in the utility.
        :return: SQL WHERE clause as a string
        """
        if not cls._season_range:
            cls._initialize_conditions()
        return "WHERE " + " AND ".join(cls._season_range) if cls._season_range else ""

