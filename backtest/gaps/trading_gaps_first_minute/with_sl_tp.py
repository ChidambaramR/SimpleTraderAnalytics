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
        self.top_n = args.get('top_n', 5)

    def get_trade_plan(self, all_stocks, daily_data, prev_date, current_date):
        # Find all stocks with a gap above threshold
        gaps = []
        for stock in all_stocks:
            stock_data = daily_data[stock]

            # Previous day and Current day data
            day_slice = stock_data[
                (stock_data.index.date == prev_date.date()) | 
                (stock_data.index.date == current_date.date())
                ]
            
            # Check if we have enough data
            if len(day_slice) < 2:
                continue

            prev_close = day_slice.iloc[0]['close']
            current_open = day_slice.iloc[1]['open']
            gap_percent = ((current_open - prev_close) / prev_close) * 100
            if abs(gap_percent) >= self.gap_threshold:
                gaps.append({
                    'symbol': stock,
                    'gap_percent': gap_percent,
                    'abs_gap_percent': abs(gap_percent),
                    'prev_close': prev_close,
                    'open': current_open
                })

        # Sort by absolute gap and take top N
        gaps.sort(key=lambda x: x['abs_gap_percent'], reverse=True)
        selected = gaps[:self.top_n]
        if not selected:
            return {}
        
        # Allocate capital equally (5x leverage)
        per_stock_capital = (self.current_capital * 5) / len(selected)
        plan = {g['symbol']: per_stock_capital for g in selected}
        return plan

    def generate_trades(self, stock, day_data, minute_data, available_capital):
        if minute_data is None or minute_data.empty or len(minute_data) < 2 or len(day_data) < 2:
            return []

        prev_close = day_data.iloc[0]['close']
        current_open = day_data.iloc[1]['open']
        gap_percent = ((current_open - prev_close) / prev_close) * 100
        quantity = floor(available_capital / current_open)
        if quantity == 0:
            return []
        exit_minute_time = datetime.strptime(self.exit_time, '%H:%M').time()
        market_open_time = datetime.strptime('09:15', '%H:%M').time()
        minutes_to_exit = (
            (exit_minute_time.hour - market_open_time.hour) * 60 + 
            (exit_minute_time.minute - market_open_time.minute)
        )
        trade = {
            'Symbol': stock,
            'Date': minute_data.index[0],
            'Entry Minute': 1,
            'Entry price': current_open,
            'Quantity': quantity,
            'Position': 'SHORT' if gap_percent > 0 else 'LONG',
            'Gap %': gap_percent
        }
        if gap_percent > 0:  # Gap Up - Short
            stop_loss_price = current_open * (1 + self.stop_loss_pct/100)
            take_profit_price = current_open * (1 - self.take_profit_pct/100)
            for i in range(0, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                if float(current_minute['high']) >= stop_loss_price:
                    trade['Exit price'] = stop_loss_price
                    trade['Exit reason'] = 'Stop Loss'
                    trade['Exit Minute'] = i
                    break
                elif float(current_minute['low']) <= take_profit_price:
                    trade['Exit price'] = take_profit_price
                    trade['Exit reason'] = 'Take Profit'
                    trade['Exit Minute'] = i
                    break
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = f'Time Exit ({self.exit_time})'
                    trade['Exit Minute'] = i
                    break
        else:  # Gap Down - Long
            stop_loss_price = current_open * (1 - self.stop_loss_pct/100)
            take_profit_price = current_open * (1 + self.take_profit_pct/100)
            for i in range(0, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                if float(current_minute['low']) <= stop_loss_price:
                    trade['Exit price'] = stop_loss_price
                    trade['Exit reason'] = 'Stop Loss'
                    trade['Exit Minute'] = i
                    break
                elif float(current_minute['high']) >= take_profit_price:
                    trade['Exit price'] = take_profit_price
                    trade['Exit reason'] = 'Take Profit'
                    trade['Exit Minute'] = i
                    break
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = f'Time Exit ({self.exit_time})'
                    trade['Exit Minute'] = i
                    break
        if 'Exit price' not in trade:
            return []
        entry_value = quantity * current_open
        exit_value = quantity * trade['Exit price']
        gross_pnl = exit_value - entry_value if trade['Position'] == 'LONG' else entry_value - exit_value
        fees = calculate_fees(entry_value, exit_value, trade['Position'])
        trade.update({
            'Entry Value': entry_value,
            'Exit Value': exit_value,
            'Fees': fees['total'],
            'PNL': gross_pnl - fees['total']
        })
        return [trade]

def run_backtest(from_date, to_date, initial_capital=100000, args={}):
    results = FirstMinuteGapTrader(initial_capital=initial_capital, args=args).run_backtest(from_date, to_date)
    return {
        'total_trades': results['total_trades'],
        'win_ratio': results['win_ratio'],
        'initial_capital': results['initial_capital'],
        'capital_added': results['capital_added'],
        'final_capital': results['final_capital'],
        'profit': results['profit'],
        'max_drawdown': results['max_drawdown'],
        'roi': results['roi'],
        'avg_profit_per_trade': results['avg_profit_per_trade'],
        'avg_loss_per_trade': results['avg_loss_per_trade'],
        'avg_daily_profit': results['avg_daily_profit'],
        'avg_daily_loss': results['avg_daily_loss'],
        'stock_stats': results['stock_stats'],
    }