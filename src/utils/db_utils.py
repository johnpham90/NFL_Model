import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

def get_connection():
    """Create a connection to the Supabase database using environment variables"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database URL and modify it for SQLAlchemy
        DATABASE_URL = os.getenv('DATABASE_URL').replace('postgres://', 'postgresql://')
        
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        print('Successfully connected to the database!')
        return engine
    except Exception as error:
        print(f'Error connecting to the database: {error}')
        return None

def execute_query(query, params=None):
    """Execute a query and return results as a pandas DataFrame"""
    engine = None
    try:
        engine = get_connection()
        if engine is None:
            return None
            
        # SQLAlchemy 2.0+ requires text() wrapper for raw SQL
        with engine.connect() as conn:
            if params:
                df = pd.read_sql_query(text(query), conn, params=params)
            else:
                df = pd.read_sql_query(text(query), conn)
        return df
    except Exception as error:
        print(f'Error executing query: {error}')
        import traceback
        traceback.print_exc()
        return None
    finally:
        if engine:
            engine.dispose()