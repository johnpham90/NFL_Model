import requests
import datetime as dt
from typing import Dict, List

def get_current_nfl_week() -> Dict:
    """
    Fetch the current NFL week and its games from the SportsDataIO API.
    Returns the current week number and lists of home teams, away teams, and game schedules.
    """
    # API URL
    api_url = "https://api.sportsdata.io/v3/nfl/scores/json/Schedules/2024?key=604a48fab9784f9fb2d6101874bec4bb"
    
    try:
        # Fetch the schedule data
        response = requests.get(api_url)
        response.raise_for_status()
        schedule_data = response.json()
        
        # Get today's date
        today = dt.date.today()
        
        # Initialize variables
        current_week = None
        home_teams: List[str] = []
        away_teams: List[str] = []
        game_schedule: List[Dict] = []
        
        for game in schedule_data:
            # Safely get the 'Date' field
            game_date_str = game.get('Date')
            if not game_date_str:  # Skip games with missing or None 'Date'
                print(f"Skipping game due to missing Date: {game}")
                continue
            
            try:
                # Parse the game date
                game_date = dt.datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S").date()
            except ValueError:
                print(f"Skipping game due to invalid Date format: {game_date_str}")
                continue
            
            # Identify the current week
            week = game.get('Week')
            if game_date <= today <= game_date + dt.timedelta(days=6):
                current_week = week
                # Filter games on Thursday (3), Saturday (5), Sunday (6), or Monday (0)
                if game_date.weekday() in [3, 5, 6, 0]:
                    home_team = game.get('HomeTeam', 'Unknown')
                    away_team = game.get('AwayTeam', 'Unknown')
                    
                    # Add to lists
                    home_teams.append(home_team)
                    away_teams.append(away_team)
                    game_schedule.append({
                        'away_team': away_team,
                        'home_team': home_team,
                        'date': game_date_str
                    })
        print({
            'current_week': current_week,
            'home_teams': home_teams,
            'away_teams': away_teams,
            'game_schedule': game_schedule
        })

        # Return the current week, home teams, away teams, and schedule
        return {
            'current_week': current_week,
            'home_teams': home_teams,
            'away_teams': away_teams,
            'game_schedule': game_schedule
        }


    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {'current_week': None, 'home_teams': [], 'away_teams': [], 'game_schedule': []}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'current_week': None, 'home_teams': [], 'away_teams': [], 'game_schedule': []}