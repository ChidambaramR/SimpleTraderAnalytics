import logging
import sqlite3
import pandas as pd
from datetime import datetime
import os

def total_gaps(from_date, to_date, interval):
    """
    Query gaps statistics and return as a DataFrame.
    Analyzes gaps > 2% across all stocks in the database.
    """
    try:
        # Construct the correct path to the database file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # get queries.py directory
        db_path = os.path.join(current_dir, '..', 'ohlc_data', f'{interval}.db')
        
        print(f"Attempting to connect to database at: {db_path}")  # Debug print
        
        conn = sqlite3.connect(db_path)
        
        # Get all table names (stocks) from the database
        table_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """
        tables = pd.read_sql_query(table_query, conn)
        
        # Convert dates to datetime format
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        print(f"Tables: {tables}")
        print(f"Querying from {from_date} to {to_date}")

        all_gaps = []
        total_trading_days = 0  # To track total trading days
        
        # Process each stock table
        for table in tables['name']:
            logging.debug(f"Processing table: {table}")
            
            query = f"""
            SELECT ts, open, close 
            FROM "{table}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            ORDER BY ts
            """
            
            df = pd.read_sql_query(query, conn, params=(from_date, to_date))
            total_trading_days += len(df)  # Add to total trading days
            logging.debug(f"Found {len(df)} records for {table}")
            
            if len(df) > 0:
                # Calculate gaps
                df['prev_close'] = df['close'].shift(1)
                df['gap_percent'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
                
                # Identify gap types
                df['gap_type'] = None
                df.loc[df['gap_percent'] > 2, 'gap_type'] = 'Gap Up'
                df.loc[df['gap_percent'] < -2, 'gap_type'] = 'Gap Down'
                
                gaps = df[df['gap_type'].notna()]
                if len(gaps) > 0:
                    logging.debug(f"Found {len(gaps)} gaps in {table}")
                all_gaps.append(gaps)
        
        conn.close()
        
        if all_gaps and total_trading_days > 0:
            combined_gaps = pd.concat(all_gaps, ignore_index=True)
            
            total_gaps_count = len(combined_gaps)
            gap_up_count = len(combined_gaps[combined_gaps['gap_type'] == 'Gap Up'])
            gap_down_count = len(combined_gaps[combined_gaps['gap_type'] == 'Gap Down'])
            
            # Calculate percentages
            total_gaps_percentage = (total_gaps_count / total_trading_days) * 100
            gap_up_percentage = (gap_up_count / total_gaps_count) * 100 if total_gaps_count > 0 else 0
            gap_down_percentage = (gap_down_count / total_gaps_count) * 100 if total_gaps_count > 0 else 0
            
            chart_data = {
                'labels': ['Total Gaps', 'Gap Up', 'Gap Down'],
                'values': [total_gaps_count, gap_up_count, gap_down_count],
                'percentages': {
                    'total': f"{total_gaps_percentage:.2f}% of trading days",
                    'up': f"{gap_up_percentage:.2f}% of total gaps",
                    'down': f"{gap_down_percentage:.2f}% of total gaps"
                }
            }
        else:
            chart_data = {
                'labels': ['Total Gaps', 'Gap Up', 'Gap Down'],
                'values': [0, 0, 0],
                'percentages': {
                    'total': "0% of trading days",
                    'up': "0% of total gaps",
                    'down': "0% of total gaps"
                }
            }
        
        return chart_data
        
    except Exception as e:
        return {'error': f"Error processing gaps: {str(e)}"} 