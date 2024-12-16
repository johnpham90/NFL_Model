import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.db_utils import get_connection, execute_query

def process_weekly_games_with_h2h() -> List[Dict]:
    """
    Process each week's games and get the last 5 head-to-head matchups for teams playing.
    """
    query = """
    SELECT 
        awayteamid,
        hometeamid,
        awayscore,
        homescore,
        date,
        week,
        season
    FROM stats.gamesummary
    ORDER BY date ASC
    """
    
    try:
        # Get all games
        all_games = execute_query(query)
        
        dtype = [
            ('away_team', 'U10'),
            ('home_team', 'U10'),
            ('away_score', 'int32'),
            ('home_score', 'int32'),
            ('date', 'datetime64[D]'),
            ('week', 'int32'),
            ('season', 'U4')
        ]
        
        games = np.array(all_games, dtype=dtype)
        
        # Get current season games
        current_season = '2024'
        season_games = games[games['season'] == current_season]
        weeks = np.unique(season_games['week'])
        
        results = []
        
        # Process each week
        for week in weeks:
            week_games = season_games[season_games['week'] == week]
            
            # Process each game in the week
            for game in week_games:
                away_team = game['away_team']
                home_team = game['home_team']
                game_date = game['date']
                
                # Get last 5 h2h games before this game
                h2h_mask = (
                    ((games['away_team'] == away_team) & (games['home_team'] == home_team)) |
                    ((games['away_team'] == home_team) & (games['home_team'] == away_team))
                ) & (games['date'] < game_date)
                
                h2h_games = games[h2h_mask]
                h2h_games = np.sort(h2h_games, order='date')[::-1][:5]  # Last 5 games
                
                # Calculate h2h stats
                h2h_stats = {
                    'season': current_season,
                    'week': int(week),
                    'date': str(game_date),
                    'away_team': away_team,
                    'home_team': home_team,
                    'game_result': {
                        'away_score': int(game['away_score']),
                        'home_score': int(game['home_score'])
                    },
                    'h2h_matchups': {
                        'total_games': len(h2h_games),
                        'previous_games': []
                    }
                }
                
                if len(h2h_games) > 0:
                    away_wins = 0
                    home_wins = 0
                    ties = 0
                    
                    for prev_game in h2h_games:
                        # Determine scores from perspective of current away/home teams
                        if prev_game['away_team'] == away_team:
                            away_score = prev_game['away_score']
                            home_score = prev_game['home_score']
                        else:
                            away_score = prev_game['home_score']
                            home_score = prev_game['away_score']
                        
                        # Record result
                        if away_score > home_score:
                            away_wins += 1
                        elif home_score > away_score:
                            home_wins += 1
                        else:
                            ties += 1
                            
                        # Add game details
                        h2h_stats['h2h_matchups']['previous_games'].append({
                            'date': str(prev_game['date']),
                            'season': prev_game['season'],
                            'away_score': int(away_score),
                            'home_score': int(home_score),
                            'result': 'away_win' if away_score > home_score else 
                                    'home_win' if home_score > away_score else 'tie'
                        })
                    
                    h2h_stats['h2h_matchups'].update({
                        'away_wins': away_wins,
                        'home_wins': home_wins,
                        'ties': ties
                    })
                
                results.append(h2h_stats)
                
        return results
        print(results)
    except Exception as e:
        print(f"Error processing games: {str(e)}")
        return []
