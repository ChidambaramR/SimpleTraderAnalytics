from math import floor
import pandas as pd
import numpy as np
from database.utils.db_utils import get_db_and_tables
from ...utils.runner import run_backtest_with_cache
from ...utils.fee_calculator import calculate_fees

def _run_backtest_with_amount(from_date, to_date, initial_capital, top_n=5):
    """
    Run backtest with fixed position sizing.
    
    Args:
        initial_capital: Starting capital (pre-leverage)
        top_n: Number of top gapped stocks to trade
    """
    try:
        conn, tables = get_db_and_tables('day')
        
        # Initialize tracking variables
        trades = []
        total_trades = 0
        wins = 0
        current_drawdown = 0
        max_drawdown = 0
        peak_equity = initial_capital
        current_equity = initial_capital  # Start with initial capital
        
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
            daily_gaps = []

            for table in tables['name']:
                query = f"""
                SELECT ts, open, close 
                FROM "{table}"
                WHERE date(ts) IN (?, ?)
                ORDER BY ts
                """
                
                df = pd.read_sql_query(query, conn, params=(current_date, next_date))
                
                if len(df) >= 2:
                    prev_close = df.iloc[0]['close']
                    current_open = df.iloc[1]['open']
                    current_close = df.iloc[1]['close']
                    
                    # Calculate gap percentage
                    gap_percent = ((current_open - prev_close) / prev_close) * 100
                    
                    if abs(gap_percent) >= 3:
                        daily_gaps.append({
                            'date': df.iloc[1]['ts'],
                            'symbol': table,
                            'open': current_open,
                            'close': current_close,
                            'prev_close': prev_close,
                            'gap_percent': gap_percent,
                            'abs_gap_percent': abs(gap_percent)
                        })
            
            if daily_gaps:
                daily_gaps.sort(key=lambda x: x['abs_gap_percent'], reverse=True)
                selected_gaps = daily_gaps[:top_n]
                
                available_capital = initial_capital * 5
                per_stock_amount = available_capital / len(selected_gaps)
                
                daily_pnl = 0
                
                for gap in selected_gaps:
                    trade = gap.copy()
                    trade['invested_amount'] = per_stock_amount
                    
                    # Calculate quantity (floor to avoid fractional shares)
                    quantity = floor(per_stock_amount / trade['open'])
                    trade['quantity'] = quantity
                    
                    print(f"\nTrading {gap['symbol']}")
                    print(f"Per stock amount: {per_stock_amount:.2f}")
                    print(f"Stock price: {trade['open']:.2f}")
                    print(f"Quantity: {quantity}")
                    print(f"Actual invested: {quantity * trade['open']:.2f}")
                    
                    # Calculate P&L
                    if gap['gap_percent'] > 0:  # Gap Up - Short
                        trade['direction'] = 'SHORT'
                        entry_value = quantity * trade['open']
                        exit_value = quantity * trade['close']
                        gross_pnl = entry_value - exit_value
                        
                        # Calculate and store fees
                        fees = calculate_fees(entry_value, exit_value, 'SHORT')
                        trade['fees'] = fees['total']
                        trade['pnl'] = gross_pnl - fees['total']
                        
                        print(f"Gap %: {gap['gap_percent']:.2f}%")
                        print(f"Entry: {trade['open']:.2f} x {quantity} = {entry_value:.2f}")
                        print(f"Exit: {trade['close']:.2f} x {quantity} = {exit_value:.2f}")
                        print(f"Gross P&L: {gross_pnl:.2f}")
                        print(f"Fees: {fees['total']:.2f}")
                        print(f"Net P&L: {trade['pnl']:.2f}")
                    
                    else:  # Gap Down - Long
                        trade['direction'] = 'LONG'
                        entry_value = quantity * trade['open']
                        exit_value = quantity * trade['close']
                        gross_pnl = exit_value - entry_value
                        
                        # Calculate and store fees
                        fees = calculate_fees(entry_value, exit_value, 'LONG')
                        trade['fees'] = fees['total']
                        trade['pnl'] = gross_pnl - fees['total']
                    
                    daily_pnl += trade['pnl']
                    trades.append(trade)
                    total_trades += 1
                    if trade['pnl'] > 0:
                        wins += 1
                
                current_equity += daily_pnl
                print(f"\nDaily P&L: {daily_pnl:.2f}")
                print(f"Current Equity after trades: {current_equity:.2f}")
                
                peak_equity = max(peak_equity, current_equity)
                current_drawdown = peak_equity - current_equity
                max_drawdown = max(max_drawdown, current_drawdown)
        
        conn.close()
        
        # Ensure we have trades before calculating statistics
        if not trades:
            return {
                'total_trades': 0,
                'win_ratio': 0,
                'initial_capital': initial_capital,
                'final_equity': initial_capital,
                'profit': 0,
                'max_drawdown': 0,
                'roi': 0
            }
        
        # Calculate final statistics
        trades_df = pd.DataFrame(trades)
        win_ratio = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = current_equity - initial_capital
        roi = (total_pnl / initial_capital * 100)
        
        return {
            'total_trades': total_trades,
            'win_ratio': round(win_ratio, 2),
            'initial_capital': round(initial_capital, 2),
            'final_equity': round(current_equity, 2),
            'profit': round(total_pnl, 2),
            'max_drawdown': round(max_drawdown, 2),
            'roi': round(roi, 2)
        }
        
    except Exception as e:
        return {
            'error': f"Error in backtest: {str(e)}",
            'total_trades': 0,
            'win_ratio': 0,
            'initial_capital': initial_capital,
            'final_equity': initial_capital,
            'profit': 0,
            'max_drawdown': 0,
            'roi': 0
        }

def _run_backtest(from_date, to_date):
    """
    Run backtest for both investment amounts.
    """
    try:
        # Run for 1L
        results_1L = _run_backtest_with_amount(from_date, to_date, initial_capital=100000)
        if 'error' in results_1L:
            return results_1L
            
        # Run for 10L
        results_10L = _run_backtest_with_amount(from_date, to_date, initial_capital=1000000)
        if 'error' in results_10L:
            return results_10L
        
        # Combine results
        return {
            'total_trades': results_1L['total_trades'],  # Same for both
            'win_ratio': results_1L['win_ratio'],        # Same for both
            'initial_capital_1L': results_1L['initial_capital'],
            'initial_capital_10L': results_10L['initial_capital'],
            'final_equity_1L': results_1L['final_equity'],
            'final_equity_10L': results_10L['final_equity'],
            'profit_1L': results_1L['profit'],
            'profit_10L': results_10L['profit'],
            'max_drawdown_1L': results_1L['max_drawdown'],
            'max_drawdown_10L': results_10L['max_drawdown'],
            'roi_1L': results_1L['roi'],
            'roi_10L': results_10L['roi']
        }
        
    except Exception as e:
        return {
            'error': f"Error in backtest: {str(e)}",
            'total_trades': 0,
            'win_ratio': 0,
            'initial_capital_1L': 100000,
            'initial_capital_10L': 1000000,
            'final_equity_1L': 100000,
            'final_equity_10L': 1000000,
            'profit_1L': 0,
            'profit_10L': 0,
            'max_drawdown_1L': 0,
            'max_drawdown_10L': 0,
            'roi_1L': 0,
            'roi_10L': 0
        }

def run_backtest_fixed_position(from_date, to_date, force_run=False):
    """
    Public interface for running the backtest with caching support.
    """
    strategy_name = 'gaps_trading_daywise_without_sl_tp_fixed_position'
    return run_backtest_with_cache(
        strategy_name=strategy_name,
        from_date=from_date,
        to_date=to_date,
        backtest_func=_run_backtest,
        force_run=force_run
    )