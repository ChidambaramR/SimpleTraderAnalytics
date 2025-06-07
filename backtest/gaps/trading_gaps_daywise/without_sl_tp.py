from ...framework.day_trader import DayTrader
from math import floor
from ...utils.fee_calculator import calculate_fees

class DaywiseGapTrader(DayTrader):
    def __init__(self, initial_capital=100000, args=None):
        super().__init__(initial_capital)
        self.gap_threshold = 3  # hardcoded as in original logic
        self.top_n = (args or {}).get('top_n', 5)
        self.slippage_pct = (args or {}).get('slippage_pct', 0.1)  # default 0.1%
        self.args = args or {}

    def get_trade_plan(self, all_stocks, daily_data, prev_date, current_date):
        gaps = []
        for stock in all_stocks:
            stock_data = daily_data[stock]
            day_slice = stock_data[
                (stock_data.index.date == prev_date.date()) |
                (stock_data.index.date == current_date.date())
            ]

            if len(day_slice) < 2:
                continue

            prev_close = day_slice.iloc[0]['close']
            current_open = day_slice.iloc[1]['open']

            gap_percent = (current_open - prev_close) / prev_close * 100
            if abs(gap_percent) >= self.gap_threshold:
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

        per_stock_capital = self.current_capital / len(selected)
        plan = {g['symbol']: per_stock_capital for g in selected}

        return plan

    def generate_trades(self, stock, day_data, minute_data, available_capital):
        if len(day_data) < 2:
            return []

        prev_close = day_data.iloc[0]['close']
        current_open = day_data.iloc[1]['open']
        current_close = day_data.iloc[1]['close']

        gap_percent = (current_open - prev_close) / prev_close * 100

        quantity = floor(available_capital / current_open)
        if quantity == 0:
            return []

        direction = 'SHORT' if gap_percent > 0 else 'LONG'

        # Apply slippage
        slippage = self.slippage_pct / 100
        if direction == 'LONG':
            entry_price = current_open * (1 + slippage)
            exit_price = current_close * (1 - slippage)
        else:
            entry_price = current_open * (1 - slippage)
            exit_price = current_close * (1 + slippage)

        if direction == 'SHORT':
            pnl = (entry_price - exit_price) * quantity
        else:
            pnl = (exit_price - entry_price) * quantity

        entry_value = quantity * entry_price
        exit_value = quantity * exit_price
        fees = calculate_fees(entry_value, exit_value, direction)
        net_pnl = pnl - fees['total']

        # Determine entry_time and exit_time from minute_data if available
        if minute_data is not None and not minute_data.empty:
            entry_time = minute_data.index[0]
            exit_time = minute_data.index[-1]
        else:
            entry_time = day_data.index[1]
            exit_time = day_data.index[1]

        trade = {
            'Symbol': stock,
            'Date': day_data.index[1],
            'Entry price': entry_price,
            'Exit price': exit_price,
            'Quantity': quantity,
            'Position': direction,
            'Gap %': gap_percent,
            'PNL': net_pnl,
            'Gross PNL': pnl,
            'Fees': fees['total'],
            'Slippage %': self.slippage_pct,
            'Exit reason': 'EOD',
            'entry_time': entry_time,
            'exit_time': exit_time,
        }
        return [trade]

def run_backtest(from_date, to_date, initial_capital=100000, args={}):
    trader = DaywiseGapTrader(initial_capital=initial_capital, args=args)
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