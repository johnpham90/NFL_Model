import psycopg2
from configparser import ConfigParser

def config(filename='config/db_config.yaml', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in {filename}')
    return db

def connect():
    try:
        params = config()
        print('Connecting to PostgreSQL database...')
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
        db_version = cur.fetchone()
        print(db_version)
        return conn, cur
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None, None

def fetch_game_data(cur, season=None):
    query = '''
    SELECT * FROM nfl_games 
    WHERE season = %s
    '''
    cur.execute(query, (season,))
    return cur.fetchall()
