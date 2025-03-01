import pandas as pd
from datetime import timedelta
from database.utils.db_utils import get_db_and_tables

def get_stitched_ohlc_data(date: str, stock: str, lookback_minutes: int = 120) -> pd.DataFrame:
    """
    Fetches and stitches together OHLC data for a given date and stock.
    Gets current date's minute data and previous N minutes of data.
    
    Args:
        date (str): Date in 'YYYY-MM-DD' format
        stock (str): Stock symbol
        lookback_minutes (int): Number of previous minutes of data to fetch (default: 120)
        
    Returns:
        pd.DataFrame: DataFrame with OHLC data, with columns:
                     ['ts', 'open', 'high', 'low', 'close', 'volume']
    """
    try:
        # Convert date string to datetime
        current_date = pd.to_datetime(date)
        
        # Get minute data connections
        minute_connections = get_db_and_tables('minute')
        
        if stock not in minute_connections:
            raise ValueError(f"No minute data found for stock: {stock}")
            
        minute_conn = minute_connections[stock]
        
        try:
            # Get current date's data
            current_query = f"""
            SELECT ts, open, high, low, close, volume
            FROM "{stock}"
            WHERE date(ts) = ?
            ORDER BY ts
            """
            current_df = pd.read_sql_query(
                current_query, 
                minute_conn, 
                params=(current_date.strftime('%Y-%m-%d'),)
            )
            
            if current_df.empty:
                raise ValueError(f"No data found for {stock} on {date}")
            
            # Get previous 5 days data to handle weekends and holidays
            prev_query = f"""
            SELECT ts, open, high, low, close, volume
            FROM "{stock}"
            WHERE date(ts) BETWEEN date(?) AND date(?)
            AND time(ts) >= '09:15:00'
            ORDER BY ts DESC
            """
            prev_df = pd.read_sql_query(
                prev_query, 
                minute_conn, 
                params=(
                    (current_date - timedelta(days=5)).strftime('%Y-%m-%d'),
                    (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
                )
            )
            
            # Convert timestamp to datetime for both dataframes
            current_df['ts'] = pd.to_datetime(current_df['ts'])
            prev_df['ts'] = pd.to_datetime(prev_df['ts'])
            
            # Get the first timestamp of current date's data
            current_start_ts = current_df['ts'].min()
            
            # Filter previous data to get only lookback_minutes before current date's first entry
            prev_df = prev_df[prev_df['ts'] < current_start_ts]
            prev_df = prev_df.head(lookback_minutes)  # Take only required number of previous points
            
            # Combine the dataframes and sort
            df = pd.concat([prev_df.iloc[::-1], current_df])  # Reverse prev_df to get chronological order
            df = df.reset_index(drop=True)
            
            return df
            
        finally:
            # Close all unique connections
            for conn in set(minute_connections.values()):
                conn.close()
                
    except Exception as e:
        raise Exception(f"Error fetching OHLC data for {stock} on {date}: {str(e)}")
