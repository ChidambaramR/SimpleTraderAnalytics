import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

def get_opening_gaps_trader_stats(from_date=None, to_date=None, result_type='ANY'):
    """
    Read and process trading stats from SQLite database.
    
    Args:
        from_date: datetime object for start date (optional)
        to_date: datetime object for end date (optional)
        result_type: 'ANY', 'PROFIT', or 'LOSS'
        
    Returns:
        pandas DataFrame with filtered stats and statistics dictionary
    """
    try:
        # Read from SQLite database
        conn = sqlite3.connect('prod_stats.db')
        
        # Base query
        query = "SELECT * FROM ledger"
        
        # Add date range filter if provided
        if from_date and to_date:
            query += " WHERE date BETWEEN ? AND ?"
            combined_df = pd.read_sql_query(query, conn, params=(
                from_date.strftime('%Y-%m-%d'),
                to_date.strftime('%Y-%m-%d')
            ))
        else:
            combined_df = pd.read_sql_query(query, conn)
            
        conn.close()
        
        # Ensure date is in correct format
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
        elif 'entry_time' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['entry_time']).dt.strftime('%Y-%m-%d')
    
        # Filter for entry and exit tags
        entry_tags = ['OGTEN', 'NOO1____', 'NOO2____']
        exit_tags = ['OGTEX', 'XOO1____', 'XOO2____']
        
        filtered_df = combined_df[
            combined_df['entry_tag'].isin(entry_tags) & 
            combined_df['exit_tag'].isin(exit_tags)
        ]
        
        # Round price and PNL columns to 2 decimal places
        price_pnl_columns = [
            'entry_price', 'exit_price', 
            'gross_pnl', 'brokerage', 'net_pnl',
            'stop_loss', 'target',
            'buy_value', 'sell_value',
            'charges', 'stt', 'transaction_charges',
            'stamp_charges', 'sebi_charges'
        ]
        
        for col in price_pnl_columns:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].round(2)
        
        # Calculate statistics
        stats = {}
        
        # Overall statistics
        stats['overall_pnl'] = filtered_df['net_pnl'].sum().round(2)
        total_trades = len(filtered_df)
        profitable_trades = len(filtered_df[filtered_df['net_pnl'] > 0])
        stats['overall_win_ratio'] = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Leg1 statistics (NOO1____ or OGTEN)
        leg1_df = filtered_df[filtered_df['entry_tag'].isin(['NOO1____', 'OGTEN'])]
        stats['leg1_pnl'] = leg1_df['net_pnl'].sum().round(2)
        leg1_trades = len(leg1_df)
        leg1_profitable = len(leg1_df[leg1_df['net_pnl'] > 0])
        stats['leg1_win_ratio'] = (leg1_profitable / leg1_trades * 100) if leg1_trades > 0 else 0
        
        # Leg2 statistics (NOO2____)
        leg2_df = filtered_df[filtered_df['entry_tag'] == 'NOO2____']
        stats['leg2_pnl'] = leg2_df['net_pnl'].sum().round(2)
        leg2_trades = len(leg2_df)
        leg2_profitable = len(leg2_df[leg2_df['net_pnl'] > 0])
        stats['leg2_win_ratio'] = (leg2_profitable / leg2_trades * 100) if leg2_trades > 0 else 0
        
        # Add date range info to stats
        stats['date_range'] = 'All Time' if not from_date else f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}"
        
        # Apply profit/loss filter if specified
        if result_type == 'PROFIT':
            filtered_df = filtered_df[filtered_df['net_pnl'] > 0]
        elif result_type == 'LOSS':
            filtered_df = filtered_df[filtered_df['net_pnl'] < 0]
        
        # Sort by date and reorder columns to have date first
        filtered_df = filtered_df.sort_values('date')
        cols = ['date'] + [col for col in filtered_df.columns if col != 'date']
        filtered_df = filtered_df[cols]
        
        return filtered_df, stats
        
    except Exception as e:
        print(f"Error in get_opening_gaps_trader_stats: {str(e)}")
        return None, None 