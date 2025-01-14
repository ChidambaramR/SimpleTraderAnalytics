import sqlite3
import os
import pandas as pd

def get_db_connection(interval):
    """
    Creates and returns a database connection for the given interval.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'ohlc_data', f'{interval}.db')
        return sqlite3.connect(db_path)
    except Exception as e:
        raise Exception(f"Error connecting to database: {str(e)}")

def get_table_names(conn):
    """
    Returns all table names from the database.
    """
    try:
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """
        return pd.read_sql_query(query, conn)
    except Exception as e:
        raise Exception(f"Error getting table names: {str(e)}")

def get_db_and_tables(interval):
    """
    Returns both database connection and table names.
    """
    try:
        conn = get_db_connection(interval)
        tables = get_table_names(conn)
        return conn, tables
    except Exception as e:
        raise Exception(f"Error in database setup: {str(e)}") 