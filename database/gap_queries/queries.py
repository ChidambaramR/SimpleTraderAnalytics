import logging
import pandas as pd
from datetime import datetime
from ..utils.db_utils import get_db_and_tables

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