from math import floor
import pandas as pd
import numpy as np
from database.utils.db_utils import get_db_and_tables
from ...utils.runner import run_backtest_with_cache
from ...utils.fee_calculator import calculate_fees
from ...framework.day_trader import DayTrader

class GapTraderFixedPosition(DayTrader):
    def __init__(self, initial_capital, top_n=5):
        super().__init__(initial_capital)
        self.top_n = top_n
        self.daily_gaps = []
    
    def reset_state_for_next_day(self):
        self.daily_gaps = []
    
    def should_trade_stock(self, day_data, stock_name):
        if len(day_data) < 2:
            return False
            
        prev_close = day_data.iloc[0]['close']
        current_open = day_data.iloc[1]['open']
        
        gap_percent = ((current_open - prev_close) / prev_close) * 100
        
        if abs(gap_percent) >= 3 and abs(gap_percent) <= 7:
            self.daily_gaps.append({
                'symbol': stock_name,
                'gap_percent': gap_percent,
                'abs_gap_percent': abs(gap_percent),
                'prev_close': prev_close,
                'open': current_open,
                'close': day_data.iloc[1]['close']
            })
            return True
        return False
    
    def get_available_capital(self, tradeable_stocks):
        if not tradeable_stocks:
            return {}
        per_stock_capital = (self.current_equity * 5) / len(tradeable_stocks)  # Using 5x leverage
        return {stock: per_stock_capital for stock in tradeable_stocks}
    
    def generate_trades(self, stock, day_data, minute_data, available_capital):
        # Sort gaps and select top N
        if len(self.daily_gaps) >= self.top_n:
            self.daily_gaps.sort(key=lambda x: x['abs_gap_percent'], reverse=True)
            self.daily_gaps = self.daily_gaps[:self.top_n]
            
            # Check if current stock is in top N
            if not any(gap['symbol'] == stock for gap in self.daily_gaps):
                return []
        
        # Get current stock's gap info
        gap = next(gap for gap in self.daily_gaps if gap['symbol'] == stock)
        
        # Calculate quantity using provided capital
        quantity = floor(available_capital / gap['open'])
        
        # Generate trade
        trade = {
            'Date': day_data.index[1],
            'Current cash available': self.current_equity,
            'Stock': stock,
            'Prev close': gap['prev_close'],
            'Current open': gap['open'],
            'Current close': gap['close'],
            'Gap type': 'SHORT' if gap['gap_percent'] > 0 else 'LONG',
            'Gap pct (absolute)': gap['abs_gap_percent'],
            'Invested amount': available_capital,
            'Quantity': quantity
        }
        
        # Calculate P&L
        if gap['gap_percent'] > 0:  # Gap Up - Short
            entry_value = quantity * gap['open']
            exit_value = quantity * gap['close']
            gross_pnl = entry_value - exit_value
            fees = calculate_fees(entry_value, exit_value, 'SHORT')
        else:  # Gap Down - Long
            entry_value = quantity * gap['open']
            exit_value = quantity * gap['close']
            gross_pnl = exit_value - entry_value
            fees = calculate_fees(entry_value, exit_value, 'LONG')
        
        trade.update({
            'Entry Value': entry_value,
            'Exit Value': exit_value,
            'Fees': fees['total'],
            'PNL': gross_pnl - fees['total']
        })
        
        return [trade]

def _run_backtest(from_date, to_date, args={}):
    """Run backtest for both investment amounts."""
    try:
        # Run for 1L
        trader_1L = GapTraderFixedPosition(initial_capital=100000)
        results_1L = trader_1L.run_backtest(from_date, to_date)
        if 'error' in results_1L:
            return results_1L
            
        # Run for 10L
        trader_10L = GapTraderFixedPosition(initial_capital=1000000)
        results_10L = trader_10L.run_backtest(from_date, to_date)
        if 'error' in results_10L:
            return results_10L
        
        # Combine and return results
        return {
            'total_trades': results_1L['total_trades'],
            'win_ratio': results_1L['win_ratio'],
            'initial_capital_1L': results_1L['initial_capital'],
            'initial_capital_10L': results_10L['initial_capital'],
            'final_equity_1L': results_1L['final_equity'],
            'final_equity_10L': results_10L['final_equity'],
            'profit_1L': results_1L['profit'],
            'profit_10L': results_10L['profit'],
            'max_drawdown_1L': results_1L['max_drawdown'],
            'max_drawdown_10L': results_10L['max_drawdown'],
            'roi_1L': results_1L['roi'],
            'roi_10L': results_10L['roi'],
            'avg_profit_per_trade_1L': results_1L['avg_profit_per_trade'],
            'avg_profit_per_trade_10L': results_10L['avg_profit_per_trade'],
            'avg_loss_per_trade_1L': results_1L['avg_loss_per_trade'],
            'avg_loss_per_trade_10L': results_10L['avg_loss_per_trade'],
            'avg_daily_profit_1L': results_1L['avg_daily_profit'],
            'avg_daily_profit_10L': results_10L['avg_daily_profit'],
            'avg_daily_loss_1L': results_1L['avg_daily_loss'],
            'avg_daily_loss_10L': results_10L['avg_daily_loss'],
            'percentile_90_profit_per_trade_1L': results_1L['percentile_90_profit_per_trade'],
            'percentile_90_profit_per_trade_10L': results_10L['percentile_90_profit_per_trade'],
            'percentile_90_loss_per_trade_1L': results_1L['percentile_90_loss_per_trade'],
            'percentile_90_loss_per_trade_10L': results_10L['percentile_90_loss_per_trade'],
            'percentile_90_daily_profit_1L': results_1L['percentile_90_daily_profit'],
            'percentile_90_daily_profit_10L': results_10L['percentile_90_daily_profit'],
            'percentile_90_daily_loss_1L': results_1L['percentile_90_daily_loss'],
            'percentile_90_daily_loss_10L': results_10L['percentile_90_daily_loss']
        }
    except Exception as e:
        return {'error': f"Error in backtest: {str(e)}"}

def run_backtest_fixed_position(from_date, to_date, args={}):
    """Public interface for running the backtest with caching support."""
    strategy_name = 'gaps_trading_daywise_without_sl_tp_fixed_position'
    return run_backtest_with_cache(
        strategy_name=strategy_name,
        from_date=from_date,
        to_date=to_date,
        backtest_func=_run_backtest
    )