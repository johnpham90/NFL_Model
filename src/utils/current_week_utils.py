import requests
import datetime as dt
from typing import Dict

def get_current_nfl_week() -> Dict:
    """
    Fetch the current NFL week and its games from the SportsDataIO API.
    Returns the current week number and games scheduled for that week.
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
        current_week_games = []
        
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
            week = game.get('Week')  # Safely get the 'Week' field
            if game_date <= today <= game_date + dt.timedelta(days=6):
                current_week = week
                # Filter games on Thursday (3), Saturday (5), Sunday (6), or Monday (0)
                if game_date.weekday() in [3, 5, 6, 0]:
                    current_week_games.append({
                        'away_team': game.get('AwayTeam', 'Unknown'),
                        'home_team': game.get('HomeTeam', 'Unknown'),
                        'date': game_date_str,
                        'week': week,
                        'season': game.get('SeasonType', 'Unknown')
                    })
        
        # Return the current week and its games
        return {
            'current_week': current_week,
            'games': current_week_games
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {'current_week': None, 'games': []}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'current_week': None, 'games': []}