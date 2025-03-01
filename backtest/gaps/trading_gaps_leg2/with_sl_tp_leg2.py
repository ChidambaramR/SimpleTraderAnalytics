from math import floor
from ...framework.day_trader import DayTrader
from ...utils.fee_calculator import calculate_fees
from ...utils.runner import run_backtest_with_cache
from datetime import datetime

class Leg2GapTrader(DayTrader):
    def __init__(self, initial_capital, args):
        super().__init__(initial_capital)
        self.gap_threshold = args.get('gap_threshold', 3)
        self.stop_loss_pct = args.get('stop_loss_pct', 0.75)
        self.take_profit_pct = args.get('take_profit_pct', 2)
        self.entry_time = args.get('entry_time', '09:17')
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
        
        # Convert exit time to minute number (e.g., '09:16' -> 2nd minute, '09:17' -> 3rd minute)
        market_open_time = datetime.strptime('09:15', '%H:%M').time()
        entry_minute_time = datetime.strptime(self.entry_time, '%H:%M').time()
        exit_minute_time = datetime.strptime('15:15', '%H:%M').time()

        minutes_to_enter = (
            (entry_minute_time.hour - market_open_time.hour) * 60 + 
            (entry_minute_time.minute - market_open_time.minute)
        )
        minutes_to_exit = (
            (exit_minute_time.hour - market_open_time.hour) * 60 + 
            (exit_minute_time.minute - market_open_time.minute)
        )

        entry_price = minute_data.iloc[minutes_to_enter]['open']

        # Calculate position size
        quantity = floor(available_capital / entry_price)
        if quantity == 0:
            return []
        
        first_min_open = minute_data.iloc[0]['open']
        first_min_close = minute_data.iloc[0]['close']
        recent_min_close = minute_data.iloc[minutes_to_enter - 1]['close']
        abs_pct_diff = abs((recent_min_close - first_min_open) / first_min_open) * 100 # abs() is enough as there are other checks which take care of gap up/gap down
        threshold_pct = 1 # Hardcode

        # Initialize trade
        trade = {
            'Symbol': stock,
            'Date': minute_data.index[0],
            'Quantity': quantity,
            'Position': 'SHORT' if gap['gap_percent'] > 0 else 'LONG',
            'Gap %': gap['gap_percent']
        }

        sl_slippage_pct = 0.5
        tp_slippage_pct = 0.25

        # Set stop loss and take profit prices based on position type
        # Gap Up - Short
        if gap['gap_percent'] > 0 \
            and first_min_close > first_min_open and recent_min_close > first_min_open and abs_pct_diff > threshold_pct:
            stop_loss_price = entry_price * (1 + self.stop_loss_pct/100)
            sl_exit_price = entry_price * (1 + (self.stop_loss_pct + sl_slippage_pct)/100)
            take_profit_price = entry_price * (1 - self.take_profit_pct/100)
            tp_exit_price = entry_price * (1 - (self.take_profit_pct - tp_slippage_pct)/100)

            # Loop through minutes until exit time
            for i in range(minutes_to_enter, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                
                # Check for stop loss
                if float(current_minute['high']) >= stop_loss_price:
                    trade['Exit price'] = sl_exit_price
                    trade['Exit reason'] = 'Stop Loss'
                    break
                    
                # Check for take profit
                elif float(current_minute['low']) <= take_profit_price:
                    trade['Exit price'] = tp_exit_price
                    trade['Exit reason'] = 'Take Profit'
                    break
                    
                # If this is the exit minute
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = 'Time Exit With Profit' if current_minute['close'] < entry_price else 'Time Exit With Loss'
                    break

        # Gap Down - Long
        elif gap['gap_percent'] < 0 \
            and first_min_close < first_min_open and recent_min_close < first_min_open and abs_pct_diff > threshold_pct:

            stop_loss_price = entry_price * (1 - self.stop_loss_pct/100)
            sl_exit_price = entry_price * (1 - (self.stop_loss_pct + sl_slippage_pct)/100)
            take_profit_price = entry_price * (1 + self.take_profit_pct/100)
            tp_exit_price = entry_price * (1 + (self.take_profit_pct - tp_slippage_pct)/100)

            # Loop through minutes until exit time
            for i in range(minutes_to_enter, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                
                # Check for stop loss
                if float(current_minute['low']) <= stop_loss_price:
                    trade['Exit price'] = sl_exit_price
                    trade['Exit reason'] = 'Stop Loss'
                    break
                    
                # Check for take profit
                elif float(current_minute['high']) >= take_profit_price:
                    trade['Exit price'] = tp_exit_price
                    trade['Exit reason'] = 'Take Profit'
                    break
                    
                # If this is the exit minute
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = 'Time Exit With Profit' if current_minute['close'] > entry_price else 'Time Exit With Loss'
                    break
        
        # If no exit condition was met (not enough minute data)
        if 'Exit price' not in trade:
            return []
        
        # Calculate P&L
        entry_value = quantity * entry_price
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
        trader_1L = Leg2GapTrader(initial_capital=100000, args=args)
        results_1L = trader_1L.run_backtest(from_date, to_date)
        if 'error' in results_1L:
            return results_1L
            
        # Run for 10L
        trader_10L = Leg2GapTrader(initial_capital=1000000, args=args)
        results_10L = trader_10L.run_backtest(from_date, to_date)
        if 'error' in results_10L:
            return results_10L
        
        # Combine and return results
        return {
            'total_trades': results_1L['total_trades'],
            'win_ratio': results_1L['win_ratio'],
            'initial_capital_1L': results_1L['initial_capital'],
            'initial_capital_10L': results_10L['initial_capital'],
            'capital_added_1L': results_1L['capital_added'],
            'capital_added_10L': results_10L['capital_added'],
            'exit_reason_1L': results_1L['exit_reason_pct'],
            'exit_reason_10L': results_10L['exit_reason_pct'],
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

def run_gaps_with_sl_tp_leg2(from_date, to_date, force_run=False, args={}):
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