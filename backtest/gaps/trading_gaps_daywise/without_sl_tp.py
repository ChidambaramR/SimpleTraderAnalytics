import pandas as pd
from database.utils.db_utils import get_db_and_tables
from ...utils.runner import run_backtest_with_cache
from collections import defaultdict

def _run_backtest(from_date, to_date, args={}):
    """
    Internal function that implements the actual backtest logic.
    Processes data day by day to simulate real trading conditions.
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

        # Get all dates in the range
        date_query = f"""
        SELECT DISTINCT date(ts) as trade_date
        FROM "{tables['name'].iloc[0]}"
        WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
        ORDER BY trade_date
        """
        trading_days = pd.read_sql_query(date_query, conn, params=(from_date, to_date))

        # Process each trading day except the last one
        for i in range(len(trading_days) - 1):
            current_date = trading_days.iloc[i]['trade_date']
            next_date = trading_days.iloc[i + 1]['trade_date']
            
            # Dictionary to store daily trades
            daily_trades = []

            # Check each stock for gaps
            for table in tables['name']:
                # Get data for current and next trading day
                query = f"""
                SELECT ts, open, close 
                FROM "{table}"
                WHERE date(ts) IN (?, ?)
                ORDER BY ts
                """
                
                df = pd.read_sql_query(query, conn, params=(current_date, next_date))
                
                if len(df) >= 2:  # Need at least 2 days of data
                    prev_close = df.iloc[0]['close']
                    current_open = df.iloc[1]['open']
                    current_close = df.iloc[1]['close']
                    
                    # Calculate gap percentage
                    gap_percent = (current_open - prev_close) / prev_close * 100
                    
                    # Check if gap meets our criteria (>= 3% or <= -3%)
                    if abs(gap_percent) >= 3:
                        trade = {
                            'date': df.iloc[1]['ts'],
                            'symbol': table,
                            'open': current_open,
                            'close': current_close,
                            'gap_percent': gap_percent,
                            'direction': 'SHORT' if gap_percent > 0 else 'LONG'
                        }
                        
                        # Calculate P&L
                        if gap_percent > 0:  # Gap Up - Short
                            trade['pnl'] = current_open - current_close
                        else:  # Gap Down - Long
                            trade['pnl'] = current_close - current_open
                        
                        trade['pnl_percentage'] = (trade['pnl'] / current_open) * 100
                        daily_trades.append(trade)
            
            # Process daily trades
            for trade in daily_trades:
                trades.append(trade)
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
            
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Calculate final statistics
        win_ratio = (wins / total_trades * 100) if total_trades > 0 else 0
        total_invested = trades_df['open'].sum()
        total_pnl_1x = trades_df['pnl'].sum()
        total_pnl_5x = total_pnl_1x * 5  # 5x leverage
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

def run_backtest(from_date, to_date, force_run=False, args={}):
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