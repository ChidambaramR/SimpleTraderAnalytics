import sqlite3
import json
import os

def get_backtest_db_connection():
    """
    Creates and returns a connection to the backtest results database.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'data', 'backtest_results.db')
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            strategy_name TEXT,
            from_date TEXT,
            to_date TEXT,
            results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (strategy_name, from_date, to_date)
        )
        """
        conn.execute(create_table_query)
        conn.commit()
        
        return conn
    except Exception as e:
        raise Exception(f"Error connecting to backtest database: {str(e)}")

def save_backtest_results(strategy_name, from_date, to_date, results):
    """
    Saves backtest results to the database.
    """
    try:
        conn = get_backtest_db_connection()
        
        # Convert results to JSON string
        results_json = json.dumps(results)
        
        # Insert or replace existing results
        query = """
        INSERT OR REPLACE INTO backtest_results 
        (strategy_name, from_date, to_date, results)
        VALUES (?, ?, ?, ?)
        """
        
        conn.execute(query, (strategy_name, from_date, to_date, results_json))
        conn.commit()
        conn.close()
        
    except Exception as e:
        raise Exception(f"Error saving backtest results: {str(e)}")

def get_backtest_results(strategy_name, from_date, to_date):
    """
    Retrieves backtest results from the database.
    Returns None if no results found.
    """
    try:
        conn = get_backtest_db_connection()
        
        query = """
        SELECT results 
        FROM backtest_results 
        WHERE strategy_name = ? 
        AND from_date = ? 
        AND to_date = ?
        """
        
        cursor = conn.execute(query, (strategy_name, from_date, to_date))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
        
    except Exception as e:
        raise Exception(f"Error retrieving backtest results: {str(e)}") 