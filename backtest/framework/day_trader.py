from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from ..utils.fee_calculator import calculate_fees
from database.utils.db_utils import get_db_and_tables


class DayTrader(ABC):
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
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
    def should_trade_stock(self, day_data, stock_name):
        """
        Determine if we should trade this stock based on daily data.
        Returns: bool
        """
        pass
        
    @abstractmethod
    def generate_trades(self, stock, day_data, minute_data, available_capital):
        """
        Generate trades for a single stock on a single day.
        Returns: list of trade dictionaries
        """
        pass
    
    @abstractmethod
    def reset_state_for_next_day(self):
        """
        Reset state for next day
        """
        pass
    
    @abstractmethod
    def get_available_capital(self, tradeable_stocks):
        """
        Calculate available capital for each tradeable stock
        Returns: dict mapping stock to available capital
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

    def get_minute_data(self, stock, current_date):
        """
        Fetch minute data for a specific stock and date
        """
        minute_connections = get_db_and_tables('minute')
        
        try:
            if stock in minute_connections:
                minute_conn = minute_connections[stock]
                query = f"""
                SELECT ts, open, high, low, close, volume
                FROM "{stock}"
                WHERE date(ts) = ?
                ORDER BY ts
                """
                df = pd.read_sql_query(query, minute_conn, params=(current_date.date(),))
                df['ts'] = pd.to_datetime(df['ts'])
                df.set_index('ts', inplace=True)
                return df
            return None
            
        finally:
            for conn in set(minute_connections.values()):
                conn.close()
    
    def run_backtest(self, from_date, to_date, stock_list=None):
        try:
            # Load daily data into memory
            trading_days, daily_data = self.load_data(from_date, to_date, stock_list)
            
            # Process each trading day
            for i in range(1, len(trading_days)):
                current_date = pd.Timestamp(trading_days.iloc[i]['trade_date'])
                prev_date = pd.Timestamp(trading_days.iloc[i-1]['trade_date'])

                # Skip if gap between trading days is >= 4 days
                if (current_date - prev_date).days >= 4:
                    print(f"Skipping day {current_date} due to gap of {(current_date - prev_date).days} days")
                    continue
    
                daily_pnl = 0
                
                # First pass: Find all tradeable stocks for this day
                tradeable_stocks = []
                for stock in daily_data.keys():
                    try:
                        stock_data = daily_data[stock]
                        mask = (stock_data.index.date == prev_date.date()) | (stock_data.index.date == current_date.date())
                        day_slice = stock_data[mask]
                        
                        if len(day_slice) < 2:  # Need both days
                            continue
                            
                        if self.should_trade_stock(day_slice, stock):
                            tradeable_stocks.append(stock)
                            
                    except Exception as e:
                        print(f"Error checking stock {stock}: {str(e)}")
                        continue
                
                # Sort tradeable stocks by absolute gap percentage and take top 5
                # self.daily_gaps.sort(key=lambda x: x['abs_gap_percent'], reverse=True)
                # tradeable_stocks = [gap['symbol'] for gap in self.daily_gaps[:5]]

                # Get available capital for each stock
                capital_allocation = self.get_available_capital(tradeable_stocks)
                
                # Second pass: Generate and execute trades only for tradeable stocks
                for stock in tradeable_stocks:
                    try:
                        # Get daily data
                        stock_data = daily_data[stock]
                        mask = (stock_data.index.date == prev_date.date()) | (stock_data.index.date == current_date.date())
                        day_slice = stock_data[mask]
                        
                        # Get minute data only for tradeable stocks
                        minute_slice = self.get_minute_data(stock, current_date)
                        
                        # Generate and process trades with allocated capital
                        day_trades = self.generate_trades(stock, day_slice, minute_slice, capital_allocation[stock])
                        
                        for trade in day_trades:
                            self.trades.append(trade)
                            self.trade_logs.append(trade)
                            self.total_trades += 1
                            if trade['PNL'] > 0:
                                self.wins += 1
                            daily_pnl += trade['PNL']
                            # Update stock-level statistics
                            self.update_stock_stats(trade)
                            
                    except Exception as e:
                        print(f"Error executing trades for stock {stock}: {str(e)}")
                        continue
                
                # Update equity and drawdown
                self.current_equity += daily_pnl
                self.reset_state_for_next_day()
                self.peak_equity = max(self.peak_equity, self.current_equity)
                current_drawdown = self.peak_equity - self.current_equity
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

                if self.current_equity < self.initial_capital:
                    diff = self.initial_capital - self.current_equity
                    self.capital_added += diff
                    self.current_equity = self.initial_capital
            
            return self.get_results()
            
        except Exception as e:
            return {'error': f"Error in backtest: {str(e)}"}
    
    def get_results(self):
        """Calculate and return final results"""
        if not self.trades:
            return self.get_empty_results()
            
        trades_df = pd.DataFrame(self.trades)
        win_ratio = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_capital = self.initial_capital + self.capital_added
        total_pnl = self.current_equity - total_capital
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
        
        percentile_90_profit = profitable_trades['PNL'].quantile(0.9) if len(profitable_trades) > 0 else 0
        percentile_90_loss = abs(loss_trades['PNL'].quantile(0.1)) if len(loss_trades) > 0 else 0
        
        # Calculate daily statistics
        df_trades = pd.DataFrame(self.trade_logs)
        df_trades['Date'] = pd.to_datetime(df_trades['Date'])
        daily_pnl = df_trades.groupby('Date')['PNL'].sum()
        
        profitable_days = daily_pnl[daily_pnl > 0]
        loss_days = daily_pnl[daily_pnl <= 0]
        
        avg_daily_profit = profitable_days.mean() if len(profitable_days) > 0 else 0
        avg_daily_loss = loss_days.mean() if len(loss_days) > 0 else 0
        
        percentile_90_daily_profit = profitable_days.quantile(0.9) if len(profitable_days) > 0 else 0
        percentile_90_daily_loss = abs(loss_days.quantile(0.1)) if len(loss_days) > 0 else 0

        # Calculate stock-level metrics
        stock_metrics = self.calculate_stock_level_metrics()
        
        return {
            'total_trades': self.total_trades,
            'win_ratio': round(win_ratio, 2),
            'initial_capital': round(self.initial_capital, 2),
            'capital_added': self.capital_added,
            'exit_reason_pct': f'SL: {round(100 * exit_reason_sl_count / len(trades_df), 2)}%, TP: {round(100 * exit_reason_tp_count / len(trades_df), 2)}%, EOD Profit: {round(100 * exit_reason_eod_profit_count / len(trades_df), 2)}%, EOD Loss: {round(100 * exit_reason_eod_loss_count / len(trades_df), 2)}%',
            'final_equity': round(self.current_equity, 2),
            'profit': round(total_pnl, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'roi': round(roi, 2),
            'avg_profit_per_trade': round(avg_profit, 2),
            'avg_loss_per_trade': round(avg_loss, 2),
            'percentile_90_profit_per_trade': round(percentile_90_profit, 2),
            'percentile_90_loss_per_trade': round(percentile_90_loss, 2),
            'avg_daily_profit': round(avg_daily_profit, 2),
            'avg_daily_loss': round(avg_daily_loss, 2),
            'percentile_90_daily_profit': round(percentile_90_daily_profit, 2),
            'percentile_90_daily_loss': round(percentile_90_daily_loss, 2),
            'stock_stats': stock_metrics
        }
    
    def get_empty_results(self):
        """Return empty results with initial values"""
        return {
            'total_trades': 0,
            'win_ratio': 0,
            'initial_capital': self.initial_capital,
            'capital_added': self.capital_added,
            'final_equity': self.initial_capital,
            'profit': 0,
            'max_drawdown': 0,
            'roi': 0,
            'avg_profit_per_trade': 0,
            'avg_loss_per_trade': 0,
            'percentile_90_profit_per_trade': 0,
            'percentile_90_loss_per_trade': 0,
            'avg_daily_profit': 0,
            'avg_daily_loss': 0,
            'percentile_90_daily_profit': 0,
            'percentile_90_daily_loss': 0,
            'stock_stats': {
                'stocks': []
            }
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
        
        # Sort by absolute PNL to show both winners and losers
        stocks_df['Abs PNL'] = stocks_df['Total PNL'].abs()
        sorted_stocks = stocks_df.nlargest(self.top_n, 'Abs PNL')
        sorted_stocks = sorted_stocks.drop('Abs PNL', axis=1)  # Remove the temporary column
        
        return {
            'stocks': sorted_stocks.to_dict(orient='records')
        }

    def set_top_n_stocks(self, n):
        """Set the number of top stocks to track in each category"""
        self.top_n = n 