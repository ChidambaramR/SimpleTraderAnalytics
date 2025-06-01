import sqlite3
import duckdb
import os
import pandas as pd

def get_db_connection(interval):
    """
    Creates and returns a database connection for the given interval.
    Supports only 'day' interval for now.
    """
    try:
        if interval == 'day':
            db_name = 'day.db'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, '..', 'ohlc_data', db_name)
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
    Returns database connection(s) and table names.
    For day interval: returns (connection, tables DataFrame)
    For minute interval: returns dict mapping stock names to their connections
    """
    try:
        if interval == 'day':
            conn = get_db_connection(interval)
            tables = get_table_names(conn)
            return conn, tables
        
        else:
            raise ValueError(f"Unsupported interval in get_db_and_tables(): {interval}")
            
    except Exception as e:
        raise Exception(f"Error in database setup: {str(e)}")

def get_duckdb_minute_connection():
    """
    Returns a DuckDB connection for minute data, which is generated from parquert files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', 'ohlc_data', 'merged_parquet.duckdb')
    duckdb_conn = duckdb.connect(db_path)
    return duckdb_conn
