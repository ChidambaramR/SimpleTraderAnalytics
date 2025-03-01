import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from utils.ohlc_data_utils import get_stitched_ohlc_data
from utils.ohlc_image_utils import create_ohlc_chart

def get_opening_gaps_trader_stats(from_date=None, to_date=None, result_type='ANY'):
    """
    Read and process trading stats from SQLite database.
    Also generates OHLC charts for each trade and saves them in appropriate folders.
    
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
        
        # Generate OHLC charts for each trade
        base_dir = os.path.join('prod_stats', 'data', 'images')
        profit_dir = os.path.join(base_dir, 'profit')
        loss_dir = os.path.join(base_dir, 'loss')
        
        os.makedirs(profit_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        
        # Process each trade
        for _, row in filtered_df.iterrows():
            try:
                date = row['date']
                symbol = row['symbol']
                is_profit = row['net_pnl'] > 0
                
                # Get OHLC data
                ohlc_df = get_stitched_ohlc_data(date, symbol)
                
                if ohlc_df is not None and not ohlc_df.empty:
                    # Prepare data for plotting
                    plot_df = ohlc_df.copy()
                    plot_df.set_index('ts', inplace=True)
                    
                    # Rename columns to match mplfinance requirements
                    plot_df.columns = [col.capitalize() for col in plot_df.columns]
                    
                    # Generate filename with date
                    filename = f"{symbol}_{date}.png"
                    
                    # Create and save chart directly in profit or loss directory
                    target_dir = profit_dir if is_profit else loss_dir
                    
                    create_ohlc_chart(
                        plot_df,
                        target_dir,
                        filename
                    )
                    print(f"Generated chart for {symbol} on {date}")
                    
            except Exception as e:
                print(f"Error processing chart for {symbol} on {date}: {str(e)}")
                continue
        
        return filtered_df, stats
        
    except Exception as e:
        print(f"Error in get_opening_gaps_trader_stats: {str(e)}")
        return None, None 