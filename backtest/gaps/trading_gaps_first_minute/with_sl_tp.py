from math import floor
from ...framework.day_trader import DayTrader
from ...utils.fee_calculator import calculate_fees
from ...utils.runner import run_backtest_with_cache

class FirstMinuteGapTrader(DayTrader):
    def __init__(self, initial_capital, gap_threshold=3, stop_loss_pct=0.75, take_profit_pct=2):
        super().__init__(initial_capital)
        self.gap_threshold = gap_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.daily_gaps = []
    
    def reset_state_for_next_day(self):
        self.daily_gaps = []
    
    def should_trade_stock(self, day_data, stock_name):
        if len(day_data) < 2:
            return False
            
        prev_close = day_data.iloc[0]['close']
        current_open = day_data.iloc[1]['open']
        
        gap_percent = ((current_open - prev_close) / prev_close) * 100
        
        if abs(gap_percent) >= self.gap_threshold:
            self.daily_gaps.append({
                'symbol': stock_name,
                'gap_percent': gap_percent,
                'abs_gap_percent': abs(gap_percent),
                'prev_close': prev_close,
                'open': current_open
            })
            return True
        return False
    
    def get_available_capital(self, tradeable_stocks):
        if not tradeable_stocks:
            return {}
        per_stock_capital = (self.current_equity * 1) / len(tradeable_stocks)  # Using 1x leverage
        return {stock: per_stock_capital for stock in tradeable_stocks}
    
    def generate_trades(self, stock, day_data, minute_data, available_capital):
        if minute_data is None or len(minute_data) < 1:
            return []  # Need minute data for this strategy
            
        # Get first minute data
        first_minute = minute_data.iloc[0]
        
        # Get gap info
        gap = next(gap for gap in self.daily_gaps if gap['symbol'] == stock)
        
        # Calculate quantity using provided capital
        quantity = floor(available_capital / gap['open'])
        
        # Initialize trade
        trade = {
            'Date': minute_data.index[0],
            'Stock': stock,
            'Current cash available': self.current_equity,
            'Gap percent': gap['gap_percent'],
            'Entry price': gap['open'],
            'Exit price': first_minute['close'],
            'Quantity': quantity,
            'Trade type': 'SHORT' if gap['gap_percent'] > 0 else 'LONG'
        }
        
        # Calculate stop loss and take profit prices
        if gap['gap_percent'] > 0:  # Gap Up - Short
            stop_loss_price = gap['open'] * (1 + self.stop_loss_pct/100)
            take_profit_price = gap['open'] * (1 - self.take_profit_pct/100)
            
            # Check if stop loss or take profit was hit during the first minute
            if first_minute['high'] >= stop_loss_price:
                trade['Exit price'] = stop_loss_price
                trade['Exit reason'] = 'Stop Loss'
            elif first_minute['low'] <= take_profit_price:
                trade['Exit price'] = take_profit_price
                trade['Exit reason'] = 'Take Profit'
            else:
                trade['Exit price'] = first_minute['close']
                trade['Exit reason'] = 'First Minute Close'
                
            # Calculate P&L
            entry_value = quantity * gap['open']
            exit_value = quantity * trade['Exit price']
            gross_pnl = entry_value - exit_value
            
        else:  # Gap Down - Long
            stop_loss_price = gap['open'] * (1 - self.stop_loss_pct/100)
            take_profit_price = gap['open'] * (1 + self.take_profit_pct/100)
            
            # Check if stop loss or take profit was hit during the first minute
            if first_minute['low'] <= stop_loss_price:
                trade['Exit price'] = stop_loss_price
                trade['Exit reason'] = 'Stop Loss'
            elif first_minute['high'] >= take_profit_price:
                trade['Exit price'] = take_profit_price
                trade['Exit reason'] = 'Take Profit'
            else:
                trade['Exit price'] = first_minute['close']
                trade['Exit reason'] = 'First Minute Close'
                
            # Calculate P&L
            entry_value = quantity * gap['open']
            exit_value = quantity * trade['Exit price']
            gross_pnl = exit_value - entry_value
        
        # Calculate fees and final PNL
        fees = calculate_fees(
            entry_value, 
            exit_value, 
            'SHORT' if gap['gap_percent'] > 0 else 'LONG'
        )
        
        trade.update({
            'Entry Value': entry_value,
            'Exit Value': exit_value,
            'Fees': fees['total'],
            'PNL': gross_pnl - fees['total']
        })
        
        return [trade]

def _run_backtest(from_date, to_date):
    """Run backtest for both investment amounts."""
    try:
        # Run for 1L
        trader_1L = FirstMinuteGapTrader(initial_capital=100000)
        results_1L = trader_1L.run_backtest(from_date, to_date)
        if 'error' in results_1L:
            return results_1L
            
        # Run for 10L
        trader_10L = FirstMinuteGapTrader(initial_capital=1000000)
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

def run_gaps_first_minute_with_sl_tp(from_date, to_date, force_run=False):
    """Public interface for running the backtest with caching support."""
    strategy_name = 'gaps_trading_first_minute_with_sl_tp'
    return run_backtest_with_cache(
        strategy_name=strategy_name,
        from_date=from_date,
        to_date=to_date,
        backtest_func=_run_backtest,
        force_run=force_run
    ) 