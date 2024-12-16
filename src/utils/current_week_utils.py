import requests
import datetime as dt
from typing import List, Dict

def get_current_nfl_week():
    """
    Fetch the current NFL week and its games from the SportsDataIO API.
    
    Args:
        api_key (str): Your SportsDataIO API key.
    
    Returns:
        Dict: Current week number and games scheduled for that week.
    """
    api_url = f"https://api.sportsdata.io/v3/nfl/scores/json/Schedules/2024?key=604a48fab9784f9fb2d6101874bec4bb"
    
    try:
        # Fetch the schedule data
        response = requests.get(api_url)
        response.raise_for_status()
        schedule_data = response.json()
        
        # Get the current system date
        today = dt.date.today()
        
        # Find the current week
        current_week = None
        current_week_games = []
        for game in schedule_data:
            game_date = dt.datetime.strptime(game['Date'], "%Y-%m-%dT%H:%M:%S").date()
            week = game['Week']
            
            # Check if the game is within this week
            if game_date <= today and today <= game_date + dt.timedelta(days=6):
                current_week = week
                current_week_games.append({
                    'away_team': game['AwayTeam'],
                    'home_team': game['HomeTeam'],
                    'date': game['Date'],
                    'week': game['Week'],
                    'season': game['SeasonType']
                })
        
        # Filter games by day of the week
        filtered_games = [
            game for game in current_week_games
            if dt.datetime.strptime(game['date'], "%Y-%m-%dT%H:%M:%S").weekday() in [3, 5, 6, 0]  # Thursday, Saturday, Sunday, Monday
        ]
        
        return {
            'current_week': current_week,
            'games': filtered_games
        }

    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {'current_week': None, 'games': []}