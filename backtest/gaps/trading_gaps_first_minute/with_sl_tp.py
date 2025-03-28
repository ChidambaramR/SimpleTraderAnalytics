from math import floor
from ...framework.day_trader import DayTrader
from ...utils.fee_calculator import calculate_fees
from ...utils.runner import run_backtest_with_cache
from datetime import datetime

class FirstMinuteGapTrader(DayTrader):
    def __init__(self, initial_capital, args):
        super().__init__(initial_capital)
        self.gap_threshold = args.get('gap_threshold', 3)
        self.stop_loss_pct = args.get('stop_loss_pct', 0.75)
        self.take_profit_pct = args.get('take_profit_pct', 2)
        self.exit_time = args.get('exit_time', '09:16')
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
        per_stock_capital = (self.current_equity * 5) / len(tradeable_stocks)  # Using 5x leverage
        return {stock: per_stock_capital for stock in tradeable_stocks}
    
    def generate_trades(self, stock, day_data, minute_data, available_capital):
        """Generate trades for a given stock on a given day."""
        if minute_data is None or minute_data.empty or len(minute_data) < 2:
            return []
        
        # Find the gap for this stock
        gap = next((g for g in self.daily_gaps if g['symbol'] == stock), None)
        if not gap:
            return []
        
        # Calculate position size
        quantity = floor(available_capital / gap['open'])
        if quantity == 0:
            return []
        
        # Get first minute data
        first_minute = minute_data.iloc[0]
        
        # Convert exit time to minute number (e.g., '09:16' -> 2nd minute, '09:17' -> 3rd minute)
        exit_minute_time = datetime.strptime(self.exit_time, '%H:%M').time()
        market_open_time = datetime.strptime('09:15', '%H:%M').time()
        minutes_to_exit = (
            (exit_minute_time.hour - market_open_time.hour) * 60 + 
            (exit_minute_time.minute - market_open_time.minute)
        )
        
        # Initialize trade
        trade = {
            'Symbol': stock,
            'Date': minute_data.index[0],
            'Entry Minute': 1,
            'Entry price': gap['open'],
            'Quantity': quantity,
            'Position': 'SHORT' if gap['gap_percent'] > 0 else 'LONG',
            'Gap %': gap['gap_percent']
        }
        
        # Set stop loss and take profit prices based on position type
        if gap['gap_percent'] > 0:  # Gap Up - Short
            stop_loss_price = gap['open'] * (1 + self.stop_loss_pct/100)
            take_profit_price = gap['open'] * (1 - self.take_profit_pct/100)
            
            # Loop through minutes until exit time
            for i in range(0, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                
                # Check for stop loss
                if float(current_minute['high']) >= stop_loss_price:
                    trade['Exit price'] = stop_loss_price
                    trade['Exit reason'] = 'Stop Loss'
                    trade['Exit Minute'] = i
                    break
                    
                # Check for take profit
                elif float(current_minute['low']) <= take_profit_price:
                    trade['Exit price'] = take_profit_price
                    trade['Exit reason'] = 'Take Profit'
                    trade['Exit Minute'] = i
                    break
                    
                # If this is the exit minute
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = f'Time Exit ({self.exit_time})'
                    trade['Exit Minute'] = i
                    break
                    
        else:  # Gap Down - Long
            stop_loss_price = gap['open'] * (1 - self.stop_loss_pct/100)
            take_profit_price = gap['open'] * (1 + self.take_profit_pct/100)
            
            # Loop through minutes until exit time
            for i in range(0, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                
                # Check for stop loss
                if float(current_minute['low']) <= stop_loss_price:
                    trade['Exit price'] = stop_loss_price
                    trade['Exit reason'] = 'Stop Loss'
                    trade['Exit Minute'] = i
                    break
                    
                # Check for take profit
                elif float(current_minute['high']) >= take_profit_price:
                    trade['Exit price'] = take_profit_price
                    trade['Exit reason'] = 'Take Profit'
                    trade['Exit Minute'] = i
                    break
                    
                # If this is the exit minute
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = f'Time Exit ({self.exit_time})'
                    trade['Exit Minute'] = i
                    break
        
        # If no exit condition was met (not enough minute data)
        if 'Exit price' not in trade:
            return []
        
        # Calculate P&L
        entry_value = quantity * gap['open']
        exit_value = quantity * trade['Exit price']
        gross_pnl = exit_value - entry_value if trade['Position'] == 'LONG' else entry_value - exit_value
        
        # Calculate fees and final PNL
        fees = calculate_fees(entry_value, exit_value, trade['Position'])
        
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
        trader_1L = FirstMinuteGapTrader(initial_capital=100000, args=args)
        results_1L = trader_1L.run_backtest(from_date, to_date)
        if 'error' in results_1L:
            return results_1L
            
        # Run for 10L
        trader_10L = FirstMinuteGapTrader(initial_capital=1000000, args=args)
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

def run_gaps_first_minute_with_sl_tp(from_date, to_date, force_run=False, args={}):
    """Public interface for running the backtest with caching support."""
    strategy_name = 'gaps_trading_first_minute_with_sl_tp'
    return run_backtest_with_cache(
        strategy_name=strategy_name,
        from_date=from_date,
        to_date=to_date,
        backtest_func=_run_backtest,
        force_run=force_run,
        args=args
    ) 