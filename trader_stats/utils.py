import os
import pandas as pd
from datetime import datetime, timedelta

def get_opening_gaps_trader_stats(from_date, to_date, result_type='ANY'):
    """
    Read and process trading stats from CSV files.
    
    Args:
        from_date: datetime object for start date
        to_date: datetime object for end date
        result_type: 'ANY', 'PROFIT', or 'LOSS'
        
    Returns:
        pandas DataFrame with filtered stats
    """
    # List to store all dataframes
    dfs = []
    
    # Get all dates in range
    current_date = from_date
    while current_date <= to_date:
        # Construct file path
        file_path = os.path.join(
            'prod_stats',
            'data',
            current_date.strftime('%Y'),
            current_date.strftime('%m'),
            current_date.strftime('%d'),
            'ledger.csv'
        )
        
        try:
            # Read CSV if exists
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print("Columns in CSV:", df.columns.tolist())  # Debug print
                # Convert date column to datetime and then to string format
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                elif 'entry_time' in df.columns:  # Try alternative column name
                    df['date'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d')
                dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
        
        current_date += timedelta(days=1)
    
    if not dfs:
        return None
        
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print("Sample dates:", combined_df['date'].head())  # Debug print
    
    # Filter for entry and exit tags
    entry_tags = ['OGTEN', 'NOO1____', 'NOO2____']
    exit_tags = ['OGTEX', 'XOO1____', 'XOO2____']
    
    filtered_df = combined_df[
        combined_df['entry_tag'].isin(entry_tags) & 
        combined_df['exit_tag'].isin(exit_tags)
    ]
    
    # Apply profit/loss filter if specified
    if result_type == 'PROFIT':
        filtered_df = filtered_df[filtered_df['net_pnl'] > 0]
    elif result_type == 'LOSS':
        filtered_df = filtered_df[filtered_df['net_pnl'] < 0]
    
    return filtered_df 