import os
import json
import pandas as pd
from datetime import datetime
import re

def get_pre_market_ticks_data(date, stock):
    """
    Reads pre-market ticks data for a given date and stock, and returns it as a formatted DataFrame.
    
    Args:
        date: datetime object or string in format 'YYYY-MM-DD'
        stock: stock symbol (e.g., 'BATAINDIA')
        
    Returns:
        pandas DataFrame with formatted tick data
    """
    try:
        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
            
        # Construct file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            current_dir, 
            'data',
            date.strftime('%Y'),
            date.strftime('%m'),
            date.strftime('%d'),
            f'pre_market_ticks_data_{stock}.txt'
        )
        
        # Read and parse the file
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    # Replace datetime strings with ISO format strings
                    line = re.sub(
                        r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
                        lambda m: f'"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T{int(m.group(4)):02d}:{int(m.group(5)):02d}:{int(m.group(6)):02d}"',
                        line
                    )
                    
                    # Replace Timestamp strings
                    line = re.sub(
                        r'Timestamp\([\'"]([^\'"]+)[\'"]\)',
                        r'"\1"',
                        line
                    )
                    
                    # Parse the line and get first element of tuple
                    line_dict = eval(line)[0]  # Get first element since it's a single-element tuple
                    
                    # Convert datetime strings back to datetime objects
                    if 'last_trade_time' in line_dict:
                        line_dict['last_trade_time'] = datetime.fromisoformat(line_dict['last_trade_time'])
                    if 'exchange_timestamp' in line_dict:
                        line_dict['exchange_timestamp'] = datetime.fromisoformat(line_dict['exchange_timestamp'])
                        
                    data.append(line_dict)
                    
                except Exception as e:
                    print(f"Error parsing line: {line[:100]}...")
                    print(f"Parse Error: {str(e)}")
                    continue
                    
        if not data:
            print(f"No valid data found in file: {file_path}")
            return None, None
            
        # print(data)
        
        # Let's see what we're working with
        # print("Sample data entry:")
        # print(type(data[0]))  # Print type of first entry
        # print(data[0])  # Print first entry
        
        # Convert tuples to dictionaries if needed
        if isinstance(data[0], tuple):
            data = [dict(d) for d in data]
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create OHLC columns
        df['open'] = df['ohlc'].apply(lambda x: x.get('open'))
        df['high'] = df['ohlc'].apply(lambda x: x.get('high'))
        df['low'] = df['ohlc'].apply(lambda x: x.get('low'))
        df['close'] = df['ohlc'].apply(lambda x: x.get('close'))
        
        # Create depth columns
        for i in range(5):
            # Buy depth
            df[f'buy_depth_{i+1}'] = df['depth'].apply(
                lambda x: x.get('buy')[i] if x.get('buy') and i < len(x.get('buy')) else None
            )
            
            # Sell depth
            df[f'sell_depth_{i+1}'] = df['depth'].apply(
                lambda x: x.get('sell')[i] if x.get('sell') and i < len(x.get('sell')) else None
            )
        
        # Create summary data
        summary = {
            'last_price': df['last_price'].iloc[0],
            'last_traded_quantity': df['last_traded_quantity'].iloc[0],
            'average_traded_price': df['average_traded_price'].iloc[0],
            'last_trade_time': df['last_trade_time'].iloc[0] if 'last_trade_time' in df.columns else None,
            'high': df['high'].iloc[0],
            'low': df['low'].iloc[0],
            'close': df['close'].iloc[0]
        }
        
        # Keep only necessary columns
        columns = [
            'exchange_timestamp',
            'open',
            'total_buy_quantity',
            'total_sell_quantity',
        ]
        
        # Add depth columns
        depth_columns = (
            [f'buy_depth_{i+1}' for i in range(5)] + 
            [f'sell_depth_{i+1}' for i in range(5)]
        )
        columns.extend(depth_columns)
        
        # Add color coding for open values
        def get_color_code(current, previous):
            if current == previous:
                return ''
            
            try:
                pct_change = ((current - previous) / previous) * 100 if previous != 0 else 0
                
                if pct_change < -3:
                    return 'danger'  # darkest red
                elif pct_change < -1:
                    return 'warning'  # medium red
                elif pct_change < 0:
                    return 'light-danger'  # light red
                elif pct_change > 3:
                    return 'success'  # darkest green
                elif pct_change > 1:
                    return 'info'  # medium green
                else:
                    return 'light-success'  # light green
            except:
                return ''  # Return empty string if calculation fails
        
        # Calculate color codes for open values
        df['open_color'] = ''  # Initialize with empty string
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('open_color')] = get_color_code(
                df.iloc[i]['open'],
                df.iloc[i-1]['open']
            )
        
        # Add open_color to columns list
        columns.append('open_color')
        
        # Select columns and sort by time
        df = df[columns].copy()
        
        # Rename exchange_timestamp to ts
        df = df.rename(columns={'exchange_timestamp': 'ts'})
        
        # Sort by timestamp
        df = df.sort_values('ts')
        
        return df, summary
        
    except FileNotFoundError:
        print(f"No pre-market ticks data found for {stock} on {date}")
        return None, None
    except Exception as e:
        print(f"Error processing pre-market ticks data: {str(e)}")
        print(f"File path: {file_path}")
        return None, None

def prepare_depth_data(row):
    """
    Prepares market depth data organized by price points.
    Returns a DataFrame with buy and sell information at each price level.
    Market orders (depth_5) are added as a separate row at price 0.
    """
    # Initialize lists to store all price points and their data
    price_data = []
    market_order_row = None
    
    # Get market orders (depth_5) quantities and save as separate row
    market_buy_qty = row['buy_depth_5']['quantity'] if row['buy_depth_5'] else 0
    market_sell_qty = row['sell_depth_5']['quantity'] if row['sell_depth_5'] else 0
    
    # Create market orders row but don't add to price_data yet
    if market_buy_qty > 0 or market_sell_qty > 0:
        market_order_row = {
            'price': 0,  # Market orders
            'buy_orders': row['buy_depth_5']['orders'] if row['buy_depth_5'] else 0,
            'buy_quantity': market_buy_qty,
            'sell_orders': row['sell_depth_5']['orders'] if row['sell_depth_5'] else 0,
            'sell_quantity': market_sell_qty
        }
    
    # Collect all buy depth data (excluding depth_5)
    for i in range(1, 5):  # Only include regular orders (1-4)
        buy_depth = row[f'buy_depth_{i}']
        if buy_depth:
            price_data.append({
                'price': buy_depth['price'],
                'buy_orders': buy_depth['orders'],
                'buy_quantity': buy_depth['quantity'],
                'sell_orders': 0,
                'sell_quantity': 0
            })
    
    # Collect all sell depth data (excluding depth_5)
    for i in range(1, 5):  # Only include regular orders (1-4)
        sell_depth = row[f'sell_depth_{i}']
        if sell_depth:
            # Check if price point already exists
            existing_price = next(
                (item for item in price_data if item['price'] == sell_depth['price']), 
                None
            )
            
            if existing_price:
                existing_price['sell_orders'] = sell_depth['orders']
                existing_price['sell_quantity'] = sell_depth['quantity']
            else:
                price_data.append({
                    'price': sell_depth['price'],
                    'buy_orders': 0,
                    'buy_quantity': 0,
                    'sell_orders': sell_depth['orders'],
                    'sell_quantity': sell_depth['quantity']
                })
    
    # Sort by price in descending order (excluding market orders)
    price_data.sort(key=lambda x: x['price'], reverse=True)
    
    # Add market orders row at the beginning if it exists
    if market_order_row:
        price_data.insert(0, market_order_row)
    
    # Create DataFrame
    depth_df = pd.DataFrame(price_data)
    
    # Calculate cumulative quantities
    if not depth_df.empty:
        # For buy orders - cumulate from highest price to lowest
        depth_df['cumulative_buy_quantity'] = depth_df['buy_quantity'].cumsum()
        
        # For sell orders - cumulate from lowest price to highest
        depth_df['cumulative_sell_quantity'] = depth_df['sell_quantity'][::-1].cumsum()[::-1]
    else:
        # Add empty columns if no data
        depth_df['cumulative_buy_quantity'] = 0
        depth_df['cumulative_sell_quantity'] = 0
    
    return depth_df
