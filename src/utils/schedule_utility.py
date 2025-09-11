import pandas as pd
from sqlalchemy import text
import sys
from typing import Optional
from datetime import datetime, date
from src.utils.db_utils import get_connection


class NFLSchedule:
    def __init__(self):
        self.engine = get_connection()
        if self.engine is None:
            raise Exception("Database connection failed.")
    
    def get_full_schedule(self) -> pd.DataFrame:
        """
        Get the complete current season schedule
        
        Returns:
            pd.DataFrame: Full schedule with columns: week, day, date, hometeam, awayteam
        """
        
        query = text("""
            SELECT 
                week,
                day,
                date,
                hometeamid as hometeam,
                awayteamid as awayteam
            FROM stats.currentseasonschedule
            ORDER BY week ASC, date ASC
        """)
        
        try:
            df = pd.read_sql_query(query, self.engine)
            print(f"✅ Retrieved {len(df)} total games")
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving full schedule: {e}")
            raise
    
    def get_current_week(self, include_past_days: int = 4) -> pd.DataFrame:
        """
        Get games for the current week (including recent past games)
        
        Args:
            include_past_days (int): How many days back to include (default: 4)
                                   Updated for NFL's expanded schedule including Wednesday games
            
        Returns:
            pd.DataFrame: This week's games
        """
        
        query = text("""
            SELECT 
                season,
                week,
                day,
                date,
                hometeamid,
                awayteamid,
                spread,
                spreadfavoriteteam,
                over_under
            FROM stats.currentseasonschedule
            WHERE date >= CURRENT_DATE - INTERVAL ':past_days days'
            AND date < CURRENT_DATE + INTERVAL '14 days'
            ORDER BY date ASC
        """)
        
        try:
            df = pd.read_sql_query(query, self.engine, params={"past_days": include_past_days})
            
            if df.empty:
                print("⚠️  No games found for current week")
            else:
                print(f"🎯 Found {len(df)} games this week (including {include_past_days} days back)")
                # Show the date range for clarity
                if not df.empty:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    print(f"   📅 Date range: {min_date} to {max_date}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving current week: {e}")
            raise
    
    def get_team_schedule(self, team_id: str) -> pd.DataFrame:
        """
        Get schedule for a specific team
        
        Args:
            team_id (str): Team ID (e.g., 'PHI', 'LAC', 'DAL')
            
        Returns:
            pd.DataFrame: Team's schedule with home_away and opponent columns
        """
        
        query = text("""
            SELECT 
                week,
                day,
                date,
                hometeamid as hometeam,
                awayteamid as awayteam,
                CASE 
                    WHEN hometeamid = :team_id THEN 'HOME'
                    ELSE 'AWAY'
                END as home_away,
                CASE 
                    WHEN hometeamid = :team_id THEN awayteamid
                    ELSE hometeamid
                END as opponent
            FROM stats.currentseasonschedule
            WHERE hometeamid = :team_id OR awayteamid = :team_id
            ORDER BY week ASC, date ASC
        """)
        
        try:
            df = pd.read_sql_query(query, self.engine, params={"team_id": team_id.upper()})
            
            if df.empty:
                print(f"⚠️  No schedule found for team {team_id}")
            else:
                print(f"🏈 Retrieved {len(df)} games for {team_id}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving team schedule: {e}")
            raise


# Simple functions for easy use
def get_full_schedule() -> pd.DataFrame:
    """Get complete season schedule"""
    schedule = NFLSchedule()
    return schedule.get_full_schedule()


def get_current_week(include_past_days: int = 4) -> pd.DataFrame:
    """Get this week's games (including recent past games)
    
    Default 4 days back to account for Wed/Thu/Fri/Sat/Sun/Mon/Tue NFL schedule
    """
    schedule = NFLSchedule()
    return schedule.get_current_week(include_past_days)
    
    def get_team_schedule(self, team_id: str) -> pd.DataFrame:
        """
        Get schedule for a specific team
        
        Args:
            team_id (str): Team ID (e.g., 'PHI', 'LAC', 'DAL')
            
        Returns:
            pd.DataFrame: Team's schedule with home_away and opponent columns
        """
        
        query = text("""
            SELECT 
                week,
                day,
                date,
                hometeamid as hometeam,
                awayteamid as awayteam,
                CASE 
                    WHEN hometeamid = :team_id THEN 'HOME'
                    ELSE 'AWAY'
                END as home_away,
                CASE 
                    WHEN hometeamid = :team_id THEN awayteamid
                    ELSE hometeamid
                END as opponent
            FROM stats.currentseasonschedule
            WHERE hometeamid = :team_id OR awayteamid = :team_id
            ORDER BY week ASC, date ASC
        """)
        
        try:
            df = pd.read_sql_query(query, self.engine, params={"team_id": team_id.upper()})
            
            if df.empty:
                print(f"⚠️  No schedule found for team {team_id}")
            else:
                print(f"🏈 Retrieved {len(df)} games for {team_id}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving team schedule: {e}")
            raise


# Simple functions for easy use
def get_full_schedule() -> pd.DataFrame:
    """Get complete season schedule"""
    schedule = NFLSchedule()
    return schedule.get_full_schedule()


def get_current_week() -> pd.DataFrame:
    """Get this week's games"""
    schedule = NFLSchedule()
    return schedule.get_current_week()


def get_team_schedule(team_id: str) -> pd.DataFrame:
    """Get schedule for specific team"""
    schedule = NFLSchedule()
    return schedule.get_team_schedule(team_id)


def main():
    """Example usage"""
    
    try:
        # Get full schedule
        print("📋 Getting full schedule...")
        full = get_full_schedule()
        if not full.empty:
            print("\n📊 Sample:")
            print(full.head().to_string(index=False))
        
        # Get current week
        print("\n🎯 Getting current week...")
        current = get_current_week()
        
        # Get team schedule
        print("\n🏈 Getting PHI schedule...")
        phi = get_team_schedule('PHI')
        
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    main()