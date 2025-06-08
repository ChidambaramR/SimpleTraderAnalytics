from math import floor
from ...framework.day_trader import DayTrader
from ...utils.fee_calculator import calculate_fees
from datetime import datetime
from math import floor
from datetime import datetime

class Leg2GapTrader(DayTrader):
    def __init__(self, initial_capital, args):
        super().__init__(initial_capital)
        self.gap_threshold = args.get('gap_threshold', 3)
        self.stop_loss_pct = args.get('stop_loss_pct', 0.75)
        self.take_profit_pct = args.get('take_profit_pct', 2)
        self.entry_time = args.get('entry_time', '09:17')
        self.trade_direction = args.get('trade_direction', 'ALL')  # 'BUY_SELL', 'SELL_BUY', 'ALL'
        self.top_n = args.get('top_n', 5)

    def get_trade_plan(self, all_stocks, daily_data, prev_date, current_date):
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

            direction_condition = False
            if self.trade_direction == 'ALL':
                direction_condition = (abs(gap_percent) >= self.gap_threshold)
            elif self.trade_direction == 'BUY_SELL':
                direction_condition = (abs(gap_percent) >= self.gap_threshold and gap_percent < 0)
            elif self.trade_direction == 'SELL_BUY':
                direction_condition = (abs(gap_percent) >= self.gap_threshold and gap_percent > 0)
            if direction_condition:
                gaps.append({
                    'symbol': stock,
                    'gap_percent': gap_percent,
                    'abs_gap_percent': abs(gap_percent),
                    'prev_close': prev_close,
                    'open': current_open
                })

        gaps.sort(key=lambda x: x['abs_gap_percent'], reverse=True)
        selected = gaps[:self.top_n]
        if not selected:
            return {}

        # Allocate capital equally (5x leverage)
        per_stock_capital = (self.current_capital * 5) / len(selected)
        plan = {g['symbol']: per_stock_capital for g in selected}
        return plan

    def generate_trades(self, stock, day_data, minute_data, available_capital):
        if len(day_data) < 2 or minute_data is None or minute_data.empty or len(minute_data) < 2:
            return []

        prev_close = day_data.iloc[0]['close']
        current_open = day_data.iloc[1]['open']
        gap_percent = ((current_open - prev_close) / prev_close) * 100

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

        quantity = floor(available_capital / entry_price)
        if quantity == 0:
            return []

        first_min_open = minute_data.iloc[0]['open']
        first_min_close = minute_data.iloc[0]['close']
        recent_min_close = minute_data.iloc[minutes_to_enter - 1]['close']

        abs_pct_diff = abs((recent_min_close - first_min_open) / first_min_open) * 100
        threshold_pct = 1
        trade = {
            'Symbol': stock,
            'Date': minute_data.index[0],
            'Quantity': quantity,
            'Position': 'SHORT' if gap_percent > 0 else 'LONG',
            'Gap %': gap_percent,
            'entry_time': minute_data.index[minutes_to_enter],
        }
        exit_time = None

        sl_slippage_pct = 0.5
        tp_slippage_pct = 0.25

        if gap_percent > 0 \
            and first_min_close > first_min_open and recent_min_close > first_min_open and abs_pct_diff > threshold_pct:
            stop_loss_price = entry_price * (1 + self.stop_loss_pct/100)
            sl_exit_price = entry_price * (1 + (self.stop_loss_pct + sl_slippage_pct)/100)
            take_profit_price = entry_price * (1 - self.take_profit_pct/100)
            tp_exit_price = entry_price * (1 - (self.take_profit_pct - tp_slippage_pct)/100)

            for i in range(minutes_to_enter, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                if float(current_minute['high']) >= stop_loss_price:
                    trade['Exit price'] = sl_exit_price
                    trade['Exit reason'] = 'Stop Loss'
                    exit_time = minute_data.index[i]
                    break
                elif float(current_minute['low']) <= take_profit_price:
                    trade['Exit price'] = tp_exit_price
                    trade['Exit reason'] = 'Take Profit'
                    exit_time = minute_data.index[i]
                    break
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = 'Time Exit With Profit' if current_minute['close'] < entry_price else 'Time Exit With Loss'
                    exit_time = minute_data.index[i]
                    break
        elif gap_percent < 0 \
            and first_min_close < first_min_open and recent_min_close < first_min_open and abs_pct_diff > threshold_pct:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct/100)
            sl_exit_price = entry_price * (1 - (self.stop_loss_pct + sl_slippage_pct)/100)
            take_profit_price = entry_price * (1 + self.take_profit_pct/100)
            tp_exit_price = entry_price * (1 + (self.take_profit_pct - tp_slippage_pct)/100)

            for i in range(minutes_to_enter, minutes_to_exit):
                current_minute = minute_data.iloc[i]
                if float(current_minute['low']) <= stop_loss_price:
                    trade['Exit price'] = sl_exit_price
                    trade['Exit reason'] = 'Stop Loss'
                    exit_time = minute_data.index[i]
                    break
                elif float(current_minute['high']) >= take_profit_price:
                    trade['Exit price'] = tp_exit_price
                    trade['Exit reason'] = 'Take Profit'
                    exit_time = minute_data.index[i]
                    break
                elif i == minutes_to_exit - 1:
                    trade['Exit price'] = float(current_minute['close'])
                    trade['Exit reason'] = 'Time Exit With Profit' if current_minute['close'] > entry_price else 'Time Exit With Loss'
                    exit_time = minute_data.index[i]
                    break

        if 'Exit price' not in trade:
            return []

        if exit_time is not None:
            trade['exit_time'] = exit_time

        entry_value = quantity * entry_price
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
    trader = Leg2GapTrader(initial_capital=initial_capital, args=args)
    results = trader.run_backtest(from_date, to_date)
    trade_stats = trader.calculate_trade_level_metrics()
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
        'trade_stats': trade_stats,
    }