import logging
import pandas as pd
from datetime import datetime
from ..utils.db_utils import get_db_and_tables
import numpy as np

def analyze_daily_total_gaps(from_date, to_date):
    """
    Query daily gaps statistics and return as a DataFrame.
    Analyzes gaps > 2% across all stocks in the database.
    """
    try:
        conn, tables = get_db_and_tables('day')
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

def analyze_daily_gap_closures(from_date, to_date):
    """
    Analyzes how daily gaps close.
    """
    try:
        conn, tables = get_db_and_tables('day')
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

def analyze_daily_gap_ranges(from_date, to_date):
    """
    Analyzes daily gap distribution across different percentage ranges.
    """
    try:
        conn, tables = get_db_and_tables('day')
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        # Initialize counters for each range
        ranges = [(2,3), (3,4), (4,5), (5,6), (6,7), (7,float('inf'))]
        gap_counts = {
            'up': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": 0 for r in ranges},
            'down': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": 0 for r in ranges}
        }
        
        total_gaps_up = 0
        total_gaps_down = 0

        for table in tables['name']:
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
                
                # Count gaps in each range
                for low, high in ranges:
                    range_key = f"{low}-{high if high != float('inf') else 'above'}"
                    
                    # Gap ups
                    mask_up = (df['gap_percent'] >= low) & (df['gap_percent'] < high if high != float('inf') else True)
                    gap_counts['up'][range_key] += len(df[mask_up])
                    total_gaps_up += len(df[mask_up])
                    
                    # Gap downs
                    mask_down = (df['gap_percent'] <= -low) & (df['gap_percent'] > (-high if high != float('inf') else -float('inf')))
                    gap_counts['down'][range_key] += len(df[mask_down])
                    total_gaps_down += len(df[mask_down])
        
        conn.close()
        
        # Calculate percentages
        percentages = {
            'up': {},
            'down': {}
        }
        
        for range_key in gap_counts['up'].keys():
            percentages['up'][range_key] = (gap_counts['up'][range_key] / total_gaps_up * 100) if total_gaps_up > 0 else 0
            percentages['down'][range_key] = (gap_counts['down'][range_key] / total_gaps_down * 100) if total_gaps_down > 0 else 0
        
        return {
            'labels': list(gap_counts['up'].keys()),
            'values': {
                'up': list(percentages['up'].values()),
                'down': list(percentages['down'].values())
            },
            'details': {
                'up': gap_counts['up'],
                'down': gap_counts['down'],
                'total_up': total_gaps_up,
                'total_down': total_gaps_down
            }
        }
        
    except Exception as e:
        return {'error': f"Error analyzing gap ranges: {str(e)}"}

def analyze_daily_successful_gap_ranges(from_date, to_date):
    """
    Analyzes distribution of successful daily gaps across different percentage ranges.
    """
    try:
        conn, tables = get_db_and_tables('day')
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        # Initialize counters for each range
        ranges = [(2,3), (3,4), (4,5), (5,6), (6,7), (7,float('inf'))]
        successful_gaps = {
            'up': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": 0 for r in ranges},
            'down': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": 0 for r in ranges}
        }
        
        total_successful_up = 0
        total_successful_down = 0

        for table in tables['name']:
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
                
                # Count successful gaps in each range
                for low, high in ranges:
                    range_key = f"{low}-{high if high != float('inf') else 'above'}"
                    
                    # Successful gap ups (higher close)
                    mask_up = ((df['gap_percent'] >= low) & 
                             (df['gap_percent'] < high if high != float('inf') else True) &
                             (df['close'] > df['open']))
                    successful_gaps['up'][range_key] += len(df[mask_up])
                    total_successful_up += len(df[mask_up])
                    
                    # Successful gap downs (lower close)
                    mask_down = ((df['gap_percent'] <= -low) & 
                               (df['gap_percent'] > (-high if high != float('inf') else -float('inf'))) &
                               (df['close'] < df['open']))
                    successful_gaps['down'][range_key] += len(df[mask_down])
                    total_successful_down += len(df[mask_down])
        
        conn.close()
        
        # Calculate percentages
        percentages = {
            'up': {},
            'down': {}
        }
        
        for range_key in successful_gaps['up'].keys():
            percentages['up'][range_key] = (successful_gaps['up'][range_key] / total_successful_up * 100) if total_successful_up > 0 else 0
            percentages['down'][range_key] = (successful_gaps['down'][range_key] / total_successful_down * 100) if total_successful_down > 0 else 0
        
        return {
            'labels': list(successful_gaps['up'].keys()),
            'values': {
                'up': list(percentages['up'].values()),
                'down': list(percentages['down'].values())
            },
            'details': {
                'up': successful_gaps['up'],
                'down': successful_gaps['down'],
                'total_up': total_successful_up,
                'total_down': total_successful_down
            }
        }
        
    except Exception as e:
        return {'error': f"Error analyzing successful gap ranges: {str(e)}"}

def analyze_daily_gap_range_success_rates(from_date, to_date):
    """
    Analyzes success rate within each daily gap range.
    """
    try:
        conn, tables = get_db_and_tables('day')
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        # Initialize counters for each range
        ranges = [(2,3), (3,4), (4,5), (5,6), (6,7), (7,float('inf'))]
        gap_stats = {
            'up': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": {'total': 0, 'successful': 0} for r in ranges},
            'down': {f"{r[0]}-{r[1] if r[1] != float('inf') else 'above'}": {'total': 0, 'successful': 0} for r in ranges}
        }

        for table in tables['name']:
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
                
                # Analyze each range
                for low, high in ranges:
                    range_key = f"{low}-{high if high != float('inf') else 'above'}"
                    
                    # Gap ups in this range
                    gap_ups = df[(df['gap_percent'] >= low) & 
                               (df['gap_percent'] < high if high != float('inf') else True)]
                    gap_stats['up'][range_key]['total'] += len(gap_ups)
                    gap_stats['up'][range_key]['successful'] += len(gap_ups[gap_ups['close'] > gap_ups['open']])
                    
                    # Gap downs in this range
                    gap_downs = df[(df['gap_percent'] <= -low) & 
                                 (df['gap_percent'] > (-high if high != float('inf') else -float('inf')))]
                    gap_stats['down'][range_key]['total'] += len(gap_downs)
                    gap_stats['down'][range_key]['successful'] += len(gap_downs[gap_downs['close'] < gap_downs['open']])
        
        conn.close()
        
        # Calculate success rates for each range
        success_rates = {
            'up': {},
            'down': {}
        }
        
        for range_key in gap_stats['up'].keys():
            # Gap ups success rate
            total_up = gap_stats['up'][range_key]['total']
            success_rates['up'][range_key] = (
                gap_stats['up'][range_key]['successful'] / total_up * 100
                if total_up > 0 else 0
            )
            
            # Gap downs success rate
            total_down = gap_stats['down'][range_key]['total']
            success_rates['down'][range_key] = (
                gap_stats['down'][range_key]['successful'] / total_down * 100
                if total_down > 0 else 0
            )
        
        return {
            'labels': list(gap_stats['up'].keys()),
            'values': {
                'up': list(success_rates['up'].values()),
                'down': list(success_rates['down'].values())
            },
            'details': {
                'up': gap_stats['up'],
                'down': gap_stats['down']
            }
        }
        
    except Exception as e:
        return {'error': f"Error analyzing gap range success rates: {str(e)}"}

def analyze_first_minute_moves(from_date, to_date, analysis_time='09:15'):
    """
    Analyzes moves for gapped stocks at a specific time.
    Returns statistics about all gap movements (up/down) and their subsequent moves.
    Filters for decisive candle patterns (pure bullish/bearish).
    
    Args:
        from_date: Start date for analysis
        to_date: End date for analysis
        analysis_time: Time to analyze (format: 'HH:MM', default: '09:15')
    """
    try:
        day_conn, tables = get_db_and_tables('day')
        minute_connections = get_db_and_tables('minute')
        
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        # Calculate which minute we're analyzing
        market_open = datetime.strptime('09:15', '%H:%M').time()
        analysis_time_obj = datetime.strptime(analysis_time, '%H:%M').time()
        
        # Initialize counters and lists for statistics
        total_instances = 0
        gap_stats = {
            'gap_up': {
                'total': 0,
                'moved_up': 0,
                'moved_down': 0,
                'up_moves': [],
                'down_moves': [],
                'decisive_bearish': 0,  # For gap up moved down
                'decisive_bearish_moves': [],  # Moves for decisive bearish candles
                'decisive_bearish_high_open_diffs': []  # High to open differences for bearish candles
            },
            'gap_down': {
                'total': 0,
                'moved_up': 0,
                'moved_down': 0,
                'up_moves': [],
                'down_moves': [],
                'decisive_bullish': 0,  # For gap down moved up
                'decisive_bullish_moves': [],  # Moves for decisive bullish candles
                'decisive_bullish_low_open_diffs': []  # Low to open differences for bullish candles
            }
        }

        def is_decisive_bearish_candle(open_price, high, low, close):
            """Check if candle is decisively bearish (small upper wick, can have some lower wick)"""
            if close >= open_price:  # Not bearish
                return False
                
            body = abs(close - open_price)
            upper_wick = high - open_price
            lower_wick = close - low
            
            # Upper wick should be very small compared to body
            if upper_wick > body * 0.15:  # Allow only 15% upper wick compared to body
                return False
                
            # Lower wick can be up to 100% of body size
            if lower_wick > body:
                return False
                
            return True

        def is_decisive_bullish_candle(open_price, high, low, close):
            """Check if candle is decisively bullish (small lower wick, can have some upper wick)"""
            if close <= open_price:  # Not bullish
                return False
                
            body = abs(close - open_price)
            upper_wick = high - close
            lower_wick = open_price - low
            
            # Lower wick should be very small compared to body
            if lower_wick > body * 0.15:  # Allow only 15% lower wick compared to body
                return False
                
            # Upper wick can be up to 100% of body size
            if upper_wick > body:
                return False
                
            return True
        
        # Process each stock
        for table in tables['name']:
            # Get daily data for gap calculation
            day_query = f"""
            SELECT date(ts) as date, open, close
            FROM "{table}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            ORDER BY ts
            """
            
            daily_data = pd.read_sql_query(day_query, day_conn, params=(from_date, to_date))
            
            # Skip if not enough data
            if len(daily_data) < 2:
                continue
                
            # Calculate gaps and analyze moves
            for i in range(1, len(daily_data)):
                total_instances += 1
                prev_close = daily_data.iloc[i-1]['close']
                curr_open = daily_data.iloc[i]['open']
                curr_date = daily_data.iloc[i]['date']
                
                # Calculate gap percentage
                gap_percent = ((curr_open - prev_close) / prev_close) * 100
                
                # Skip if gap is less than 2%
                if abs(gap_percent) < 2:
                    continue
                    
                # Get minute data for the current date
                if table in minute_connections:
                    minute_query = f"""
                    SELECT time(ts) as time, open, high, low, close
                    FROM "{table}"
                    WHERE date(ts) = date(?)
                    AND time(ts) <= time(?)
                    ORDER BY ts
                    """
                    
                    minute_data = pd.read_sql_query(
                        minute_query,
                        minute_connections[table],
                        params=(curr_date, analysis_time)
                    )
                    
                    if len(minute_data) > 0:
                        # Get the last candle data
                        last_candle = minute_data.iloc[-1]
                        analysis_price = last_candle['close']
                        
                        # Calculate move percentage from open
                        move_percent = ((analysis_price - curr_open) / curr_open) * 100
                        
                        if gap_percent > 0:  # Gap Up
                            gap_stats['gap_up']['total'] += 1
                            if move_percent > 0:  # Moved further up
                                gap_stats['gap_up']['moved_up'] += 1
                                gap_stats['gap_up']['up_moves'].append(move_percent)
                            else:  # Moved down
                                gap_stats['gap_up']['moved_down'] += 1
                                gap_stats['gap_up']['down_moves'].append(abs(move_percent))
                                
                                # Check if it's a decisive bearish candle
                                if is_decisive_bearish_candle(
                                    last_candle['open'],
                                    last_candle['high'],
                                    last_candle['low'],
                                    last_candle['close']
                                ):
                                    gap_stats['gap_up']['decisive_bearish'] += 1
                                    gap_stats['gap_up']['decisive_bearish_moves'].append(abs(move_percent))
                                    # Calculate high to open difference for bearish candles
                                    high_open_diff = ((last_candle['high'] - last_candle['open']) / last_candle['open']) * 100
                                    if 'decisive_bearish_high_open_diffs' not in gap_stats['gap_up']:
                                        gap_stats['gap_up']['decisive_bearish_high_open_diffs'] = []
                                    gap_stats['gap_up']['decisive_bearish_high_open_diffs'].append(high_open_diff)
                                    
                        else:  # Gap Down
                            gap_stats['gap_down']['total'] += 1
                            if move_percent > 0:  # Moved up
                                gap_stats['gap_down']['moved_up'] += 1
                                gap_stats['gap_down']['up_moves'].append(move_percent)
                                
                                # Check if it's a decisive bullish candle
                                if is_decisive_bullish_candle(
                                    last_candle['open'],
                                    last_candle['high'],
                                    last_candle['low'],
                                    last_candle['close']
                                ):
                                    gap_stats['gap_down']['decisive_bullish'] += 1
                                    gap_stats['gap_down']['decisive_bullish_moves'].append(move_percent)
                                    # Calculate low to open difference for bullish candles
                                    low_open_diff = ((last_candle['low'] - last_candle['open']) / last_candle['open']) * 100
                                    if 'decisive_bullish_low_open_diffs' not in gap_stats['gap_down']:
                                        gap_stats['gap_down']['decisive_bullish_low_open_diffs'] = []
                                    gap_stats['gap_down']['decisive_bullish_low_open_diffs'].append(low_open_diff)
                                    
                            else:  # Moved further down
                                gap_stats['gap_down']['moved_down'] += 1
                                gap_stats['gap_down']['down_moves'].append(abs(move_percent))
        
        # Calculate statistics for moves
        for gap_type in ['gap_up', 'gap_down']:
            # Stats for upward moves
            up_moves = gap_stats[gap_type]['up_moves']
            if up_moves:
                gap_stats[gap_type]['up_avg_move'] = f"{sum(up_moves) / len(up_moves):.2f}%"
                gap_stats[gap_type]['up_std_move'] = f"{pd.Series(up_moves).std():.2f}%"
                gap_stats[gap_type]['up_p90_move'] = f"{pd.Series(up_moves).quantile(0.9):.2f}%"
            else:
                gap_stats[gap_type]['up_avg_move'] = "0.00%"
                gap_stats[gap_type]['up_std_move'] = "0.00%"
                gap_stats[gap_type]['up_p90_move'] = "0.00%"
            
            # Stats for downward moves
            down_moves = gap_stats[gap_type]['down_moves']
            if down_moves:
                gap_stats[gap_type]['down_avg_move'] = f"{sum(down_moves) / len(down_moves):.2f}%"
                gap_stats[gap_type]['down_std_move'] = f"{pd.Series(down_moves).std():.2f}%"
                gap_stats[gap_type]['down_p90_move'] = f"{pd.Series(down_moves).quantile(0.9):.2f}%"
            else:
                gap_stats[gap_type]['down_avg_move'] = "0.00%"
                gap_stats[gap_type]['down_std_move'] = "0.00%"
                gap_stats[gap_type]['down_p90_move'] = "0.00%"
            
        # Calculate statistics for decisive moves
        if gap_stats['gap_up']['decisive_bearish_moves']:
            moves = gap_stats['gap_up']['decisive_bearish_moves']
            gap_stats['gap_up']['decisive_bearish_avg'] = f"{sum(moves) / len(moves):.2f}%"
            gap_stats['gap_up']['decisive_bearish_std'] = f"{pd.Series(moves).std():.2f}%"
            gap_stats['gap_up']['decisive_bearish_p90'] = f"{pd.Series(moves).quantile(0.9):.2f}%"
            
            # Calculate high-open difference statistics for bearish candles
            high_open_diffs = gap_stats['gap_up']['decisive_bearish_high_open_diffs']
            gap_stats['gap_up']['decisive_bearish_high_open_avg'] = f"{sum(high_open_diffs) / len(high_open_diffs):.2f}%"
            gap_stats['gap_up']['decisive_bearish_high_open_std'] = f"{pd.Series(high_open_diffs).std():.2f}%"
            gap_stats['gap_up']['decisive_bearish_high_open_p90'] = f"{pd.Series(high_open_diffs).quantile(0.9):.2f}%"
        else:
            gap_stats['gap_up']['decisive_bearish_avg'] = "0.00%"
            gap_stats['gap_up']['decisive_bearish_std'] = "0.00%"
            gap_stats['gap_up']['decisive_bearish_p90'] = "0.00%"
            gap_stats['gap_up']['decisive_bearish_high_open_avg'] = "0.00%"
            gap_stats['gap_up']['decisive_bearish_high_open_std'] = "0.00%"
            gap_stats['gap_up']['decisive_bearish_high_open_p90'] = "0.00%"
            
        if gap_stats['gap_down']['decisive_bullish_moves']:
            moves = gap_stats['gap_down']['decisive_bullish_moves']
            gap_stats['gap_down']['decisive_bullish_avg'] = f"{sum(moves) / len(moves):.2f}%"
            gap_stats['gap_down']['decisive_bullish_std'] = f"{pd.Series(moves).std():.2f}%"
            gap_stats['gap_down']['decisive_bullish_p90'] = f"{pd.Series(moves).quantile(0.9):.2f}%"
            
            # Calculate low-open difference statistics for bullish candles
            low_open_diffs = gap_stats['gap_down']['decisive_bullish_low_open_diffs']
            gap_stats['gap_down']['decisive_bullish_low_open_avg'] = f"{sum(low_open_diffs) / len(low_open_diffs):.2f}%"
            gap_stats['gap_down']['decisive_bullish_low_open_std'] = f"{pd.Series(low_open_diffs).std():.2f}%"
            gap_stats['gap_down']['decisive_bullish_low_open_p90'] = f"{pd.Series(low_open_diffs).quantile(0.9):.2f}%"
        else:
            gap_stats['gap_down']['decisive_bullish_avg'] = "0.00%"
            gap_stats['gap_down']['decisive_bullish_std'] = "0.00%"
            gap_stats['gap_down']['decisive_bullish_p90'] = "0.00%"
            gap_stats['gap_down']['decisive_bullish_low_open_avg'] = "0.00%"
            gap_stats['gap_down']['decisive_bullish_low_open_std'] = "0.00%"
            gap_stats['gap_down']['decisive_bullish_low_open_p90'] = "0.00%"
        
        return {
            'total_instances': total_instances,
            'details': gap_stats
        }
        
    except Exception as e:
        return {'error': f"Error analyzing first minute moves: {str(e)}"}

def analyze_first_minute_rest_of_day_moves(from_date, to_date):
    """
    Analyzes how price moves after first minute for different gap and first minute scenarios.
    """
    try:
        day_conn, tables = get_db_and_tables('day')
        minute_connections = get_db_and_tables('minute')
        
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"
        
        # Counters for different scenarios
        gap_up_first_up = {'total': 0, 'crossed_down': 0}
        gap_up_first_down = {'total': 0, 'crossed_up': 0}
        gap_down_first_up = {'total': 0, 'crossed_down': 0}
        gap_down_first_down = {'total': 0, 'crossed_up': 0}

        for table in tables['name']:
            # Get daily data
            query = f"""
            SELECT 
                date(ts) as date,
                FIRST_VALUE(open) OVER (PARTITION BY date(ts)) as open,
                LAST_VALUE(close) OVER (PARTITION BY date(ts)) as close
            FROM "{table}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            GROUP BY date(ts)
            ORDER BY date(ts)
            """
            
            df = pd.read_sql_query(query, day_conn, params=(from_date, to_date))
            
            if len(df) > 0 and table in minute_connections:
                # Process daily data
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['prev_close'] = df['close'].shift(1)
                df['gap_percent'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
                
                # Get minute data for the entire period
                minute_query = f"""
                SELECT 
                    date(ts) as date,
                    time(ts) as time,
                    FIRST_VALUE(open) OVER (PARTITION BY date(ts) ORDER BY ts) as first_min_open,
                    FIRST_VALUE(close) OVER (PARTITION BY date(ts) ORDER BY ts) as first_min_close,
                    high,
                    low
                FROM "{table}"
                WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
                """
                minute_df = pd.read_sql_query(minute_query, minute_connections[table], 
                                            params=(from_date, to_date))
                
                if len(minute_df) > 0:
                    minute_df['date'] = pd.to_datetime(minute_df['date'])
                    
                    # Process each day
                    for date in df.index[1:]:  # Skip first day
                        gap_percent = df.loc[date, 'gap_percent']
                        day_open = df.loc[date, 'open']
                        
                        # Get this day's minute data
                        day_minute_data = minute_df[minute_df['date'] == date]
                        if len(day_minute_data) < 2:  # Need at least 2 minutes of data
                            continue
                            
                        # Get first minute data
                        first_min = day_minute_data.iloc[0]
                        first_min_open = first_min['first_min_open']
                        first_min_close = first_min['first_min_close']
                        first_min_move = ((first_min_close - first_min_open) / first_min_open) * 100
                        
                        # Get rest of day data (after 09:16)
                        rest_of_day = day_minute_data[day_minute_data['time'] > '09:16']
                        if len(rest_of_day) == 0:
                            continue
                            
                        if gap_percent > 3:  # Gap Up
                            if first_min_close > first_min_open:  # First minute up
                                gap_up_first_up['total'] += 1
                                if rest_of_day['low'].min() < day_open:
                                    gap_up_first_up['crossed_down'] += 1
                            else:  # First minute down
                                gap_up_first_down['total'] += 1
                                if rest_of_day['high'].max() > day_open:
                                    gap_up_first_down['crossed_up'] += 1
                                    
                        elif gap_percent < -3:  # Gap Down
                            if first_min_close > first_min_open:  # First minute up
                                gap_down_first_up['total'] += 1
                                if rest_of_day['low'].min() < day_open:
                                    gap_down_first_up['crossed_down'] += 1
                            else:  # First minute down
                                gap_down_first_down['total'] += 1
                                if rest_of_day['high'].max() > day_open:
                                    gap_down_first_down['crossed_up'] += 1
                    
                    del day_minute_data
                del minute_df
        
        # Close connections
        day_conn.close()
        for conn in set(minute_connections.values()):
            conn.close()
        
        return {
            'gap_up_first_up': {
                'total': gap_up_first_up['total'],
                'crossed_down': gap_up_first_up['crossed_down'],
                'crossed_percent': (gap_up_first_up['crossed_down'] / gap_up_first_up['total'] * 100) 
                                 if gap_up_first_up['total'] > 0 else 0
            },
            'gap_up_first_down': {
                'total': gap_up_first_down['total'],
                'crossed_up': gap_up_first_down['crossed_up'],
                'crossed_percent': (gap_up_first_down['crossed_up'] / gap_up_first_down['total'] * 100)
                                 if gap_up_first_down['total'] > 0 else 0
            },
            'gap_down_first_up': {
                'total': gap_down_first_up['total'],
                'crossed_down': gap_down_first_up['crossed_down'],
                'crossed_percent': (gap_down_first_up['crossed_down'] / gap_down_first_up['total'] * 100)
                                 if gap_down_first_up['total'] > 0 else 0
            },
            'gap_down_first_down': {
                'total': gap_down_first_down['total'],
                'crossed_up': gap_down_first_down['crossed_up'],
                'crossed_percent': (gap_down_first_down['crossed_up'] / gap_down_first_down['total'] * 100)
                                 if gap_down_first_down['total'] > 0 else 0
            }
        }
        
    except Exception as e:
        print(f"Error in analyze_first_minute_rest_of_day_moves: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': f"Error analyzing rest of day moves: {str(e)}"} 