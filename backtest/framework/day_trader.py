from abc import ABC, abstractmethod
import pandas as pd
from database.utils.db_utils import get_db_and_tables, get_duckdb_minute_connection


class DayTrader(ABC):
    """
    Abstract base class for simulating intraday (day trading) strategies using OHLC data at both daily and minute frequency.

    This class provides a framework for running backtests on intraday trading strategies.
    It assumes that daily and minute data is available in the database.
    The primary workflow is managed by the `run_backtest` method, which iterates over trading days 
      and stocks, simulates trades, and tracks performance metrics.

    Key Features:
    - Loads daily and minute OHLC data for a given list of stocks and date range.
    - For each trading day, performs sanity checks, slices current and previous day data, and determines tradeable stocks.
    - Uses `get_trade_plan` to select stocks to trade and allocate capital for the day.
    - Simulates trades using user-implemented logic (see `generate_trades`).
    - Tracks daily and aggregate P&L, win ratio, capital added, drawdown, and other statistics.
    - Updates equity and resets state as needed, ensuring the simulation reflects realistic capital management (e.g., topping up equity if it falls below initial capital).
    - Provides detailed results including trade logs, per-stock statistics, and summary metrics (win ratio, ROI, max drawdown, etc.).

    Abstract Methods (to be implemented by subclasses):
    - get_trade_plan(all_stocks, daily_data, prev_date, current_date): Decide which stocks to trade and how much capital to allocate to each for the current day.
    - generate_trades(stock, day_data, minute_data, available_capital): Generate trades for a stock on a given day.
    - reset_state_for_next_day(): Reset any stateful variables for the next trading day.

    Example usage:
        class MyStrategy(DayTrader):
            ... # implement abstract methods
        trader = MyStrategy(initial_capital=100000)
        results = trader.run_backtest('2023-01-01', '2023-03-31', stock_list=['AAPL', 'GOOG'])

    See also: backtest/gaps/trading_gaps_first_minute/with_sl_tp.py for an example subclass.
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.capital_added = 0
        self.peak_equity = initial_capital
        self.max_drawdown = 0
        self.trades = []
        self.trade_logs = []
        self.total_trades = 0
        self.wins = 0
        # Add stock-level tracking
        self.stock_stats = {}  # Dictionary to store per-stock statistics
        self.top_n = 10  # Default number of top stocks to track
        
    @abstractmethod
    def get_trade_plan(self, all_stocks, daily_data, prev_date, current_date):
        """
        Decide which stocks to trade and how much capital to allocate to each for the current day.
        Args:
            all_stocks: list of all stock names
            daily_data: dict of {stock: DataFrame} for all stocks
            prev_date: previous trading day (Timestamp)
            current_date: current trading day (Timestamp)
        Returns:
            dict mapping stock to capital allocation for the day
        """
        pass
        
    @abstractmethod
    def generate_trades(self, stock, day_data, minute_data, available_capital):
        """
        Generate trades for a single stock on a single day.
        Returns: list of trade dictionaries
        """
        pass

    def load_data(self, from_date, to_date, stock_list=None):
        """
        Pre-load daily data into memory
        Returns: tuple (trading_days, daily_data_dict)
        """
        # Get connections
        day_conn, day_tables = get_db_and_tables('day')
        
        try:
            # If no stock list provided, use all stocks from day.db
            if stock_list is None:
                with open('stocks_allowed_for_intraday.txt', 'r') as f:
                    stock_list = [line.strip() for line in f.readlines()]
            
            
            # Convert dates
            from_date = f"{from_date} 00:00:00"
            to_date = f"{to_date} 23:59:59"
            
            # Get all trading days
            date_query = f"""
            SELECT DISTINCT date(ts) as trade_date
            FROM "{stock_list[0]}"
            WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
            ORDER BY trade_date
            """
            trading_days = pd.read_sql_query(date_query, day_conn, params=(from_date, to_date))
            
            # Pre-load daily data for all stocks
            daily_data = {}
            for stock in stock_list:
                query = f"""
                SELECT ts, open, close 
                FROM "{stock}"
                WHERE datetime(ts) BETWEEN datetime(?) AND datetime(?)
                ORDER BY ts
                """
                df = pd.read_sql_query(query, day_conn, params=(from_date, to_date))
                df['ts'] = pd.to_datetime(df['ts'])
                df.set_index('ts', inplace=True)
                daily_data[stock] = df
            

            return trading_days, daily_data
            
        finally:
            day_conn.close()

    def get_minute_data(self, stock, current_date, duckdb_min_conn):
        """
        Fetch minute data for a specific stock and date
        """
        try:
            query = f"""
            SELECT ts, open, high, low, close, volume
            FROM "{stock}"
            WHERE date(ts) = ?
            ORDER BY ts
            """
            
            # Commenting the read query call which is recommended with sqlite3, but not recommended with DuckDB
            # df = pd.read_sql_query(query, duckdb_min_conn, params=(current_date.date(),))
            
            df = duckdb_min_conn.execute(query, (current_date.date(),)).fetchdf()
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching minute data for {stock} on {current_date}: {str(e)}")
            return None
    
    def run_backtest(self, from_date, to_date, stock_list=None):
        try:
            # Load daily data into memory
            trading_days, daily_data = self.load_data(from_date, to_date, stock_list)
            
            # Process each trading day
            duckdb_min_conn = get_duckdb_minute_connection()
            
            for i in range(1, len(trading_days)):
                current_date = pd.Timestamp(trading_days.iloc[i]['trade_date'])
                prev_date = pd.Timestamp(trading_days.iloc[i-1]['trade_date'])

                # Skip if gap between trading days is >= 4 days
                if (current_date - prev_date).days >= 4:
                    print(f"Skipping day {current_date} due to gap of {(current_date - prev_date).days} days")
                    continue

                daily_pnl = 0

                all_stocks = list(daily_data.keys())
                trade_plan = self.get_trade_plan(all_stocks, daily_data, prev_date, current_date)

                # Generate trades for each stock in the trade plan
                for stock, capital in trade_plan.items():
                    try:
                        stock_data = daily_data[stock]
                        day_slice = stock_data[
                            (stock_data.index.date == prev_date.date()) | 
                            (stock_data.index.date == current_date.date())
                            ]

                        minute_slice = self.get_minute_data(stock, current_date, duckdb_min_conn)

                        day_trades = self.generate_trades(stock, day_slice, minute_slice, capital)

                        for trade in day_trades:
                            self.trades.append(trade)
                            self.trade_logs.append(trade)
                            self.total_trades += 1
                            if trade['PNL'] > 0:
                                self.wins += 1
                            daily_pnl += trade['PNL']
                            self.update_stock_stats(trade)

                    except Exception as e:
                        print(f"Error executing trades for stock {stock}: {str(e)}")
                        continue

                # Update current cash in hand with the daily P&L
                self.current_capital += daily_pnl

                # Update peak equity and max drawdown
                self.peak_equity = max(self.peak_equity, self.current_capital)
                current_drawdown = self.peak_equity - self.current_capital
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

                # If current equity is less than initial capital, add the difference to capital added
                # This is where we top up the equity to the initial capital
                # We will track the added capital in the results
                if self.current_capital < self.initial_capital:
                    self.capital_added += (self.initial_capital - self.current_capital)
                    self.current_capital = self.initial_capital

            return self.get_results()
        except Exception as e:
            raise Exception(f"Error in backtest: {str(e)}")
        
        finally:
            duckdb_min_conn.close()
    
    def get_results(self):
        """Calculate and return final results"""
        if not self.trades:
            return self.get_empty_results()
            
        trades_df = pd.DataFrame(self.trades)
        win_ratio = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_capital = self.initial_capital + self.capital_added
        total_pnl = self.current_capital - total_capital
        roi = (total_pnl / total_capital * 100)
        
        # Calculate trade statistics
        profitable_trades = trades_df[trades_df['PNL'] > 0]
        loss_trades = trades_df[trades_df['PNL'] <= 0]

        # Find exit reason split
        exit_reason_sl_count = len(trades_df[trades_df['Exit reason'] == 'Stop Loss'])
        exit_reason_tp_count = len(trades_df[trades_df['Exit reason'] == 'Take Profit'])
        exit_reason_eod_profit_count = len(trades_df[trades_df['Exit reason'] == 'Time Exit With Profit'])
        exit_reason_eod_loss_count = len(trades_df[trades_df['Exit reason'] == 'Time Exit With Loss'])

        avg_profit = profitable_trades['PNL'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = loss_trades['PNL'].mean() if len(loss_trades) > 0 else 0
        
        # Calculate daily statistics
        df_trades = pd.DataFrame(self.trade_logs)
        df_trades['Date'] = pd.to_datetime(df_trades['Date'])
        daily_pnl = df_trades.groupby('Date')['PNL'].sum()
        
        profitable_days = daily_pnl[daily_pnl > 0]
        loss_days = daily_pnl[daily_pnl <= 0]
        
        avg_daily_profit = profitable_days.mean() if len(profitable_days) > 0 else 0
        avg_daily_loss = loss_days.mean() if len(loss_days) > 0 else 0

        # Calculate stock-level metrics
        stock_metrics = self.calculate_stock_level_metrics()
        
        return {
            'total_trades': self.total_trades,
            'win_ratio': round(win_ratio, 2),
            'initial_capital': round(self.initial_capital, 2),
            'capital_added': self.capital_added,
            'exit_reason_pct': f'SL: {round(100 * exit_reason_sl_count / len(trades_df), 2)}%, TP: {round(100 * exit_reason_tp_count / len(trades_df), 2)}%, EOD Profit: {round(100 * exit_reason_eod_profit_count / len(trades_df), 2)}%, EOD Loss: {round(100 * exit_reason_eod_loss_count / len(trades_df), 2)}%',
            'final_capital': round(self.current_capital, 2),
            'capital_added': round(self.capital_added, 2),
            'profit': round(total_pnl, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'roi': round(roi, 2),
            'avg_profit_per_trade': round(avg_profit, 2),
            'avg_loss_per_trade': round(avg_loss, 2),
            'avg_daily_profit': round(avg_daily_profit, 2),
            'avg_daily_loss': round(avg_daily_loss, 2),
            'stock_stats': stock_metrics
        }
    
    def get_empty_results(self):
        """Return empty results with initial values"""
        return {
            'total_trades': 0,
            'win_ratio': 0,
            'initial_capital': self.initial_capital,
            'capital_added': self.capital_added,
            'final_capital': self.initial_capital,
            'profit': 0,
            'max_drawdown': 0,
            'roi': 0,
            'avg_profit_per_trade': 0,
            'avg_loss_per_trade': 0,
            'avg_daily_profit': 0,
            'avg_daily_loss': 0,
            'stock_stats': {
                'stocks': []
            },
            'exit_reason_pct': 'N/A'
        }

    def update_stock_stats(self, trade):
        """Update statistics for a specific stock"""
        symbol = trade['Symbol']
        if symbol not in self.stock_stats:
            self.stock_stats[symbol] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'gap_up_trades': 0,
                'gap_up_wins': 0,
                'gap_up_losses': 0,
                'gap_down_trades': 0,
                'gap_down_wins': 0,
                'gap_down_losses': 0,
                'trades': []
            }
        
        stats = self.stock_stats[symbol]
        stats['total_trades'] += 1
        stats['total_pnl'] += trade['PNL']
        stats['trades'].append(trade)
        
        if trade['PNL'] > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
            
        # Track gap direction success
        if trade['Gap %'] > 0:  # Gap Up
            stats['gap_up_trades'] += 1
            if (trade['PNL'] > 0):
                stats['gap_up_wins'] += 1
            else:
                stats['gap_up_losses'] += 1
        else:  # Gap Down
            stats['gap_down_trades'] += 1
            if (trade['PNL'] > 0):
                stats['gap_down_wins'] += 1
            else:
                stats['gap_down_losses'] += 1

    def calculate_stock_level_metrics(self):
        """Calculate stock-level metrics and return all stocks sorted by PNL"""
        # Convert stock stats to DataFrame
        stats_list = []
        for symbol, stats in self.stock_stats.items():
            win_ratio = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            loss_ratio = (stats['losses'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            
            stats_dict = {
                'Symbol': symbol,
                'Total Trades': stats['total_trades'],
                'Wins': stats['wins'], 
                'Win Ratio': round(win_ratio, 2),
                'Losses': stats['losses'],
                'Loss Ratio': round(loss_ratio, 2),
                'Total PNL': stats['total_pnl'],
                'Gap Up Trades': stats['gap_up_trades'],
                'Gap Up Wins': stats['gap_up_wins'],
                'Gap Up Losses': stats['gap_up_losses'],
                'Gap Down Trades': stats['gap_down_trades'], 
                'Gap Down Wins': stats['gap_down_wins'],
                'Gap Down Losses': stats['gap_down_losses']
            }
            stats_list.append(stats_dict)
            
        stocks_df = pd.DataFrame(stats_list)
        
        return {
            'stocks': stocks_df.to_dict(orient='records')
        }

    def calculate_trade_level_metrics(self):
        """Return trade-level statistics for all trades with required columns."""
        trade_stats = []
        for trade in self.trades:
            entry_type = 'Buy' if trade.get('Position') == 'LONG' else 'Short'
            exit_type = 'Sell' if trade.get('Position') == 'LONG' else 'Buy'
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            gap_type = 'Up' if trade.get('Gap %', 0) > 0 else 'Down'
            trade_stats.append({
                'Symbol': trade.get('Symbol', ''),
                'PNL': trade.get('PNL', 0),
                'Entry Type': entry_type,
                'Entry Time': entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(entry_time, 'strftime') else str(entry_time),
                'Exit Type': exit_type,
                'Exit Time': exit_time,
                'Position': trade.get('Position'),
                'Gap Type': gap_type
            })
        return trade_stats