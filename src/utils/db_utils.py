import os
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Dict, List, Union

class NFLDatabaseConnector:
    def __init__(self):
        """Initialize database connection"""
        load_dotenv()
        self.engine = self._create_connection()

    def _create_connection(self):
        """Create a connection to the Supabase database"""
        try:
            DATABASE_URL = os.getenv('DATABASE_URL').replace('postgres://', 'postgresql://')
            engine = create_engine(DATABASE_URL)
            print('Successfully connected to the database!')
            return engine
        except Exception as error:
            print(f'Error connecting to the database: {error}')
            return None

    def execute_query(self, query: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as DataFrame"""
        try:
            if params:
                return pd.read_sql_query(query, self.engine, params=params)
            return pd.read_sql_query(query, self.engine)
        except Exception as error:
            print(f'Error executing query: {error}')
            return None

    # Game Data Methods
    def get_games(self, 
                 season: Optional[int] = None, 
                 team: Optional[str] = None,
                 week: Optional[int] = None,
                 home_team: Optional[str] = None,
                 away_team: Optional[str] = None) -> pd.DataFrame:
        """Get NFL game data with flexible filtering"""
        query = """
        SELECT *
        FROM nfl_games
        WHERE 1=1
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if team is not none %} AND (home_team = %(team)s OR away_team = %(team)s) {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        {% if home_team is not none %} AND home_team = %(home_team)s {% endif %}
        {% if away_team is not none %} AND away_team = %(away_team)s {% endif %}
        ORDER BY season DESC, week ASC
        """
        params = {
            'season': season,
            'team': team,
            'week': week,
            'home_team': home_team,
            'away_team': away_team
        }
        return self.execute_query(query, params)

    def get_team_stats(self,
                      season: Optional[int] = None,
                      team: Optional[str] = None,
                      week: Optional[int] = None,
                      last_n_games: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics with various filtering options"""
        query = """
        SELECT *
        FROM team_stats
        WHERE 1=1
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if team is not none %} AND team = %(team)s {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        {% if last_n_games is not none %} 
        ORDER BY game_date DESC
        LIMIT %(last_n_games)s
        {% endif %}
        """
        params = {
            'season': season,
            'team': team,
            'week': week,
            'last_n_games': last_n_games
        }
        return self.execute_query(query, params)

    def get_player_stats(self,
                        player_name: Optional[str] = None,
                        team: Optional[str] = None,
                        position: Optional[str] = None,
                        season: Optional[int] = None,
                        week: Optional[int] = None) -> pd.DataFrame:
        """Get player statistics with various filtering options"""
        query = """
        SELECT *
        FROM player_stats
        WHERE 1=1
        {% if player_name is not none %} AND player_name = %(player_name)s {% endif %}
        {% if team is not none %} AND team = %(team)s {% endif %}
        {% if position is not none %} AND position = %(position)s {% endif %}
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        ORDER BY season DESC, week ASC
        """
        params = {
            'player_name': player_name,
            'team': team,
            'position': position,
            'season': season,
            'week': week
        }
        return self.execute_query(query, params)

    def get_betting_odds(self,
                        season: Optional[int] = None,
                        week: Optional[int] = None,
                        team: Optional[str] = None) -> pd.DataFrame:
        """Get betting odds data with filters"""
        query = """
        SELECT *
        FROM betting_odds
        WHERE 1=1
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        {% if team is not none %} AND (home_team = %(team)s OR away_team = %(team)s) {% endif %}
        ORDER BY game_date DESC
        """
        params = {
            'season': season,
            'week': week,
            'team': team
        }
        return self.execute_query(query, params)

    def get_injuries(self,
                     season: Optional[int] = None,
                     week: Optional[int] = None,
                     team: Optional[str] = None,
                     player_name: Optional[str] = None) -> pd.DataFrame:
        """Get injury report data"""
        query = """
        SELECT *
        FROM injury_reports
        WHERE 1=1
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        {% if team is not none %} AND team = %(team)s {% endif %}
        {% if player_name is not none %} AND player_name = %(player_name)s {% endif %}
        ORDER BY report_date DESC
        """
        params = {
            'season': season,
            'week': week,
            'team': team,
            'player_name': player_name
        }
        return self.execute_query(query, params)

    def get_weather(self,
                    season: Optional[int] = None,
                    week: Optional[int] = None,
                    stadium: Optional[str] = None) -> pd.DataFrame:
        """Get weather data for games"""
        query = """
        SELECT *
        FROM weather_data
        WHERE 1=1
        {% if season is not none %} AND season = %(season)s {% endif %}
        {% if week is not none %} AND week = %(week)s {% endif %}
        {% if stadium is not none %} AND stadium = %(stadium)s {% endif %}
        ORDER BY game_date DESC
        """
        params = {
            'season': season,
            'week': week,
            'stadium': stadium
        }
        return self.execute_query(query, params)

    def get_tables(self) -> pd.DataFrame:
        """Get list of all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public';
        """
        return self.execute_query(query)

# Example usage:
'''
db = NFLDatabaseConnector()

# Get all games for a specific team in a season
games_df = db.get_games(season=2023, team='Chiefs')

# Get team stats for the last 5 games
recent_stats = db.get_team_stats(team='Chiefs', last_n_games=5)

# Get player stats for a specific position
qb_stats = db.get_player_stats(position='QB', season=2023)

# Get betting odds for a specific week
odds = db.get_betting_odds(season=2023, week=1)

# Get injury reports for a team
injuries = db.get_injuries(team='Chiefs', week=10)

# Get weather data for a specific stadium
weather = db.get_weather(stadium='Arrowhead Stadium')
'''
