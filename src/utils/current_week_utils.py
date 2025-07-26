import requests
import datetime as dt
from typing import Dict, List

def get_current_nfl_week() -> Dict:
    """
    Fetch the current NFL week and its games from the SportsDataIO API.
    Returns the current week number and lists of hometeamid, awayteamid, and game schedules.
    To use this Utility:
    Import Utility into file: from src.utils.current_week_utils import get_current_nfl_week
    Set a variable for get_current_nfl_week ie nfl_week = get_current_nfl_week()
    Pass in parameters in braket: awayteamid, hometeamid, game_schedule 
    
        nfl_week_data = get_current_nfl_week()
        nfl_week_data['game_schedule']
    
    """
    # API URL
    api_url = "https://api.sportsdata.io/v3/nfl/scores/json/Schedules/2024?key=604a48fab9784f9fb2d6101874bec4bb"
    
    # Team abbreviation mapping
    team_mapping = {
        'SF': 'SFO',
        'KC': 'KAN',
        'NO': 'NOR',
        'TB': 'TAM',
        'LV': 'LVR',
        'NE': 'NWE',
        'GB': 'GNB'
    }
    
    try:
        # Fetch the schedule data
        response = requests.get(api_url)
        response.raise_for_status()
        schedule_data = response.json()
        
        # Today's date and the start of the current week (Tuesday)
        today = dt.date.today()
        tuesday_of_this_week = today - dt.timedelta(days=today.weekday() - 1)  # Monday = 0, Tuesday = 1
        
        # Initialize lists
        current_week = None
        hometeamid: List[str] = []
        awayteamid: List[str] = []
        game_schedule: List[Dict] = []

        # Collect games for the current week
        for game in schedule_data:
            # Safely get the 'Date' field
            game_date_str = game.get('Date')
            if not game_date_str:
                continue
            
            try:
                game_date = dt.datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S").date()
            except ValueError:
                continue
            
            # Check if the game is in the current NFL week (Tuesday onward)
            if tuesday_of_this_week <= game_date < tuesday_of_this_week + dt.timedelta(days=7):
                week = game.get('Week')
                current_week = week  # Set the current week
                
                # Map team abbreviations and rename keys
                hometeam = team_mapping.get(game.get('HomeTeam', 'Unknown'), game.get('HomeTeam', 'Unknown'))
                awayteam = team_mapping.get(game.get('AwayTeam', 'Unknown'), game.get('AwayTeam', 'Unknown'))
                
                # Add to lists
                hometeamid.append(hometeam)
                awayteamid.append(awayteam)
                game_schedule.append({
                    'awayteamid': awayteam,
                    'hometeamid': hometeam,
                    'date': game_date_str
                })
        
        # Return the results
        return {
            'current_week': current_week,
            'hometeamid': hometeamid,
            'awayteamid': awayteamid,
            'game_schedule': game_schedule
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {'current_week': None, 'hometeamid': [], 'awayteamid': [], 'game_schedule': []}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'current_week': None, 'hometeamid': [], 'awayteamid': [], 'game_schedule': []}