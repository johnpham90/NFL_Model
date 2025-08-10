import os
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

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
    try:
        engine = get_connection()
        if params:
            df = pd.read_sql_query(text(query), engine, params=params)
        else:
            df = pd.read_sql_query(query, engine)
        return df
    except Exception as error:
        print(f'Error executing query: {error}')
        return None