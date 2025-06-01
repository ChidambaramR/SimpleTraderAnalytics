import sqlite3
import duckdb
import os
import pandas as pd
import string

def get_db_connection(interval, stock_prefix=None):
    """
    Creates and returns a database connection for the given interval.
    For minute data, requires stock_prefix to determine which db file to use.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if interval == 'minute':
            if stock_prefix is None:
                raise ValueError("stock_prefix is required for minute data")
            db_name = f"{stock_prefix}-minute.db"
        else:
            db_name = f'{interval}.db'
            
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
        
        elif interval == 'minute':
            # Define all possible prefixes (A-Z and 0-9)
            prefixes = list(string.ascii_uppercase) + ['0-9']
            
            # Create a dictionary to store connections for each stock
            stock_connections = {}
            
            # For each prefix, try to connect and get tables
            for prefix in prefixes:
                try:
                    conn = get_db_connection(interval, prefix)
                    tables = get_table_names(conn)
                    
                    # Add each table (stock) with its connection to the dictionary
                    for table in tables['name']:
                        stock_connections[table] = conn
                        
                except Exception as e:
                    # If db file doesn't exist for this prefix, skip it
                    continue
            
            return stock_connections
        
        else:
            raise ValueError(f"Unsupported interval: {interval}")
            
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
