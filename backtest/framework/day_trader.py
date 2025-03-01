from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from ..utils.fee_calculator import calculate_fees
from database.utils.db_utils import get_db_and_tables

class DayTrader(ABC):
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.max_drawdown = 0
        self.trades = []
        self.trade_logs = []
        self.total_trades = 0
        self.wins = 0
        
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
                stock_list = day_tables['name'].tolist()
            
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
                            
                    except Exception as e:
                        print(f"Error executing trades for stock {stock}: {str(e)}")
                        continue
                
                # Update equity and drawdown
                self.current_equity += daily_pnl
                self.reset_state_for_next_day()
                self.peak_equity = max(self.peak_equity, self.current_equity)
                current_drawdown = self.peak_equity - self.current_equity
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            return self.get_results()
            
        except Exception as e:
            return {'error': f"Error in backtest: {str(e)}"}
    
    def get_results(self):
        """Calculate and return final results"""
        if not self.trades:
            return self.get_empty_results()
            
        trades_df = pd.DataFrame(self.trades)
        win_ratio = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_pnl = self.current_equity - self.initial_capital
        roi = (total_pnl / self.initial_capital * 100)
        
        # Calculate trade statistics
        profitable_trades = trades_df[trades_df['PNL'] > 0]
        loss_trades = trades_df[trades_df['PNL'] <= 0]
        
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
        
        return {
            'total_trades': self.total_trades,
            'win_ratio': round(win_ratio, 2),
            'initial_capital': round(self.initial_capital, 2),
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
            'percentile_90_daily_loss': round(percentile_90_daily_loss, 2)
        }
    
    def get_empty_results(self):
        """Return empty results with initial values"""
        return {
            'total_trades': 0,
            'win_ratio': 0,
            'initial_capital': self.initial_capital,
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
            'percentile_90_daily_loss': 0
        } 