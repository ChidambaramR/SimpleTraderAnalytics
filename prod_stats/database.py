import sqlite3
import pandas as pd
import json
from datetime import datetime
import os
from utils import get_in_market_ticks_data

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('prod_stats.db')
    
    conn.commit()
    conn.close()


def store_ledger_data(df, date):
    """Store ledger data in the database"""
    conn = sqlite3.connect('prod_stats.db')
    
    # Add date column
    df['date'] = date
    
    # Store in database - will create table if it doesn't exist
    df.to_sql('ledger', conn, if_exists='append', index=False)
    
    conn.close()

def store_ticks_data(df, stock):
    """Store ticks data for a stock in the database"""
    try:
        conn = sqlite3.connect('prod_stats.db')
        
        # Create a copy to avoid modifying the original dataframe
        df_to_store = df.copy()
        
        # Convert depth data to JSON strings
        for i in range(5):
            df_to_store[f'buy_depth_{i}'] = df_to_store['depth'].apply(
                lambda x: json.dumps(x.get('buy')[i] if x.get('buy') and i < len(x.get('buy')) else None)
            )
            
            df_to_store[f'sell_depth_{i}'] = df_to_store['depth'].apply(
                lambda x: json.dumps(x.get('sell')[i] if x.get('sell') and i < len(x.get('sell')) else None)
            )
        
        # Select only the columns we want to store
        columns_to_store = [
            'ts',  # exchange_timestamp renamed to ts
            'last_price',
            'last_traded_quantity',
            'average_traded_price',
            'volume_traded',
            'total_buy_quantity',
            'total_sell_quantity',
            'change',
            'last_trade_time',
            'open',
            'high',
            'low',
            'close'
        ]
        
        # Add depth columns
        for i in range(5):
            columns_to_store.extend([f'buy_depth_{i}', f'sell_depth_{i}'])
            
        # Keep only the specified columns
        df_to_store = df_to_store[columns_to_store]
        
        # Store in database - will create table if it doesn't exist
        df_to_store.to_sql(f'ticks_{stock}', conn, if_exists='append', index=False)
        
        conn.close()
        
    except Exception as e:
        print(f"Error storing ticks data for {stock}: {str(e)}")
        raise

def rebuild_db_from_files():
    """
    Rebuilds the entire database from downloaded files.
    WARNING: This will delete the existing database and recreate it.
    """
    try:
        # Remove existing database
        if os.path.exists('prod_stats.db'):
            os.remove('prod_stats.db')
            
        # Initialize fresh database
        init_db()
        
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        
        if not os.path.exists(data_dir):
            print("No data directory found")
            return
            
        # Walk through the year/month/day directory structure
        for year in os.listdir(data_dir):
            year_path = os.path.join(data_dir, year)
            if not os.path.isdir(year_path):
                continue
                
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue
                    
                for day in os.listdir(month_path):
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path):
                        continue
                        
                    date_str = f"{year}-{month}-{day}"
                    print(f"Processing data for {date_str}")
                    
                    # Process ledger file
                    ledger_path = os.path.join(day_path, 'ledger.csv')
                    if os.path.exists(ledger_path):
                        print(f"Found ledger file for {date_str}")
                        try:
                            df = pd.read_csv(ledger_path)
                            store_ledger_data(df, date_str)
                        except Exception as e:
                            print(f"Error processing {ledger_path}: {str(e)}")
                            continue
                    
                    # Process ticks files
                    for filename in os.listdir(day_path):
                        if filename.startswith('ticks_data_'):
                            stock = filename.replace('ticks_data_', '').replace('.txt', '')
                            print(f"Processing ticks data for {stock} on {date_str}")
                            
                            try:
                                df, _ = get_in_market_ticks_data(date_str, stock)
                                if df is not None:
                                    store_ticks_data(df, stock)
                            except Exception as e:
                                print(f"Error processing {filename}: {str(e)}")
                                continue
        
        print("Database rebuild completed successfully")
        return True
        
    except Exception as e:
        print(f"Error rebuilding database: {str(e)}")
        return False 