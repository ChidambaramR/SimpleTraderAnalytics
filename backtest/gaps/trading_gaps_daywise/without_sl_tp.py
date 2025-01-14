import pandas as pd
from database.utils.db_utils import get_db_and_tables
from ...utils.runner import run_backtest_with_cache

def _run_backtest(from_date, to_date):
    """
    Internal function that implements the actual backtest logic.
    """
    try:
        conn, tables = get_db_and_tables('day')
        
        # Initialize tracking variables
        trades = []
        total_trades = 0
        wins = 0
        current_drawdown_1x = 0
        max_drawdown_1x = 0
        peak_equity_1x = 0
        current_equity_1x = 0
        
        from_date = f"{from_date} 00:00:00"
        to_date = f"{to_date} 23:59:59"

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
                
                # Filter for gaps >= 3%
                tradeable_gaps = df[abs(df['gap_percent']) >= 3].copy()
                
                for _, row in tradeable_gaps.iterrows():
                    trade = {
                        'date': row['ts'],
                        'symbol': table,
                        'open': row['open'],
                        'close': row['close'],
                        'gap_percent': row['gap_percent']
                    }
                    
                    # Calculate profit/loss
                    if row['gap_percent'] > 0:  # Gap Up - Short
                        trade['direction'] = 'SHORT'
                        trade['pnl'] = row['open'] - row['close']  # Profit if close < open
                    else:  # Gap Down - Long
                        trade['direction'] = 'LONG'
                        trade['pnl'] = row['close'] - row['open']  # Profit if close > open
                    
                    trade['pnl_percentage'] = (trade['pnl'] / row['open']) * 100
                    trades.append(trade)
                    
                    # Update statistics
                    total_trades += 1
                    if trade['pnl'] > 0:
                        wins += 1
                    
                    # Track equity and drawdown (1x leverage)
                    current_equity_1x += trade['pnl']
                    peak_equity_1x = max(peak_equity_1x, current_equity_1x)
                    current_drawdown_1x = peak_equity_1x - current_equity_1x
                    max_drawdown_1x = max(max_drawdown_1x, current_drawdown_1x)
        
        conn.close()
        
        # Ensure we have trades before calculating statistics
        if not trades:
            return {
                'total_trades': 0,
                'win_ratio': 0,
                'total_invested': 0,
                'profit_1x': 0,
                'profit_5x': 0,
                'max_drawdown_1x': 0,
                'max_drawdown_5x': 0,
                'roi_1x': 0,
                'roi_5x': 0
            }
            
        # Calculate final statistics
        win_ratio = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Calculate total invested (sum of opening prices)
        total_invested = trades_df['open'].sum()
        
        # Calculate profits
        total_pnl_1x = trades_df['pnl'].sum()
        total_pnl_5x = total_pnl_1x * 5  # 5x leverage
        
        # Calculate max drawdown with 5x leverage
        max_drawdown_5x = max_drawdown_1x * 5
        
        # Calculate ROI
        roi_1x = (total_pnl_1x / total_invested * 100) if total_invested > 0 else 0
        roi_5x = (total_pnl_5x / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_ratio': round(win_ratio, 2),
            'total_invested': round(total_invested, 2),
            'profit_1x': round(total_pnl_1x, 2),
            'profit_5x': round(total_pnl_5x, 2),
            'max_drawdown_1x': round(max_drawdown_1x, 2),
            'max_drawdown_5x': round(max_drawdown_5x, 2),
            'roi_1x': round(roi_1x, 2),
            'roi_5x': round(roi_5x, 2)
        }
        
    except Exception as e:
        # Return a dictionary with all expected fields, but with zeros
        return {
            'error': f"Error in backtest: {str(e)}",
            'total_trades': 0,
            'win_ratio': 0,
            'total_invested': 0,
            'profit_1x': 0,
            'profit_5x': 0,
            'max_drawdown_1x': 0,
            'max_drawdown_5x': 0,
            'roi_1x': 0,
            'roi_5x': 0
        }

def run_backtest(from_date, to_date, force_run=False):
    """
    Public interface for running the backtest with caching support.
    """
    strategy_name = 'gaps_trading_daywise_without_sl_tp'
    return run_backtest_with_cache(
        strategy_name=strategy_name,
        from_date=from_date,
        to_date=to_date,
        backtest_func=_run_backtest,
        force_run=force_run
    ) 