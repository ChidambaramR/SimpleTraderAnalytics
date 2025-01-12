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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'ohlc_data', f'{interval}.db')
        
        conn = sqlite3.connect(db_path)
        
        table_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """
        tables = pd.read_sql_query(table_query, conn)
        
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"

        all_gaps = []
        total_trading_days = 0
        
        for table in tables['name']:
            logging.debug(f"Processing table: {table}")
            
            query = f"""
            SELECT ts, open, close 
            FROM "{table}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            ORDER BY ts
            """
            
            df = pd.read_sql_query(query, conn, params=(from_date, to_date))
            total_trading_days += len(df)
            
            if len(df) > 0:
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

def analyze_gap_closures(from_date, to_date, interval):
    """
    Analyzes how gaps close:
    - For gap ups: % of times close price > open price
    - For gap downs: % of times close price < open price
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'ohlc_data', f'{interval}.db')
        
        conn = sqlite3.connect(db_path)
        
        table_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """
        tables = pd.read_sql_query(table_query, conn)
        
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        gap_up_higher_close = 0
        gap_up_total = 0
        gap_down_lower_close = 0
        gap_down_total = 0

        for table in tables['name']:
            logging.debug(f"Processing table: {table}")
            
            query = f"""
            SELECT ts, open, close 
            FROM "{table}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            ORDER BY ts
            """
            
            df = pd.read_sql_query(query, conn, params=(from_date, to_date))
            
            if len(df) > 0:
                df['prev_close'] = df['close'].shift(1)
                df['gap_percent'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
                
                # Identify gap types
                gap_ups = df[df['gap_percent'] > 2]
                gap_downs = df[df['gap_percent'] < -2]
                
                # For gap ups, check if close > open
                gap_up_higher_close += len(gap_ups[gap_ups['close'] > gap_ups['open']])
                gap_up_total += len(gap_ups)
                
                # For gap downs, check if close < open
                gap_down_lower_close += len(gap_downs[gap_downs['close'] < gap_downs['open']])
                gap_down_total += len(gap_downs)
        
        conn.close()
        
        # Calculate percentages
        gap_up_higher_close_pct = (gap_up_higher_close / gap_up_total * 100) if gap_up_total > 0 else 0
        gap_down_lower_close_pct = (gap_down_lower_close / gap_down_total * 100) if gap_down_total > 0 else 0
        
        return {
            'labels': ['Gap Up → Higher Close', 'Gap Down → Lower Close'],
            'values': [gap_up_higher_close_pct, gap_down_lower_close_pct],
            'details': {
                'gap_up': {
                    'total': gap_up_total,
                    'higher_close': gap_up_higher_close,
                    'percentage': f"{gap_up_higher_close_pct:.2f}%"
                },
                'gap_down': {
                    'total': gap_down_total,
                    'lower_close': gap_down_lower_close,
                    'percentage': f"{gap_down_lower_close_pct:.2f}%"
                }
            }
        }
        
    except Exception as e:
        return {'error': f"Error analyzing gap closures: {str(e)}"} 