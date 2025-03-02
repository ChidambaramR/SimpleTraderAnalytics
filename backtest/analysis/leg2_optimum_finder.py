import sqlite3
import os

import pandas as pd

def find_and_store_daily_gaps():
    """
    This function finds the daily gaps in the stock data and stores them in a database.
    This function is idempotent.
    """
    source_db = _get_db_file_path('day.db')
    output_db = _get_db_file_path('day_wise_gaps.db')

    # Create a connection to the source database
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()

    # Create a connection to the output database
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()

    # Create the output table with a unique constraint on date and stock
    output_cursor.execute('''
        CREATE TABLE IF NOT EXISTS gaps (
            date TEXT,
            stock TEXT,
            gaptype TEXT,
            pctdiff REAL,
            UNIQUE(date, stock)
        )
    ''')

    # Get the list of tables (stocks)
    source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = source_cursor.fetchall()

    # Iterate over each table
    for table in tables:
        stock = table[0]
        
        # Properly quote the table name to handle special characters
        query = f'SELECT ts, open, close FROM "{stock}" ORDER BY ts;'
        source_cursor.execute(query)
        rows = source_cursor.fetchall()

        # Iterate over the rows to find gaps
        for i in range(1, len(rows)):
            prev_close = rows[i-1][2]
            current_open = rows[i][1]
            date = rows[i][0]

            # Calculate the percentage difference
            pct_diff = ((current_open - prev_close) / prev_close) * 100

            # Check if the difference is between 3% and 8%
            if 3 <= abs(pct_diff) <= 8:
                gap_type = 'GAP_UP' if pct_diff > 0 else 'GAP_DOWN'
                # Use INSERT OR IGNORE to handle duplicates
                # Write date in YYYY-MM-DD format - the first 10 characters of the date string
                date_str = date[0:10]
                output_cursor.execute(
                    'INSERT OR IGNORE INTO gaps (date, stock, gaptype, pctdiff) VALUES (?, ?, ?, ?)',
                    (date_str, stock, gap_type, pct_diff)
                )

    # Commit the changes and close the connections
    output_conn.commit()
    source_conn.close()
    output_conn.close()

def find_optimum_trade_time_for_gapup():
    """
    This function finds the optimum trade time for gap up stocks.
    """
    output_conn, output_cursor, minute_optimal_conn, minute_optimal_cursor = _setup_database_connections('day_wise_gaps.db', 'minute_wise_optimal.db')

    # Fetch GAP_UP stocks
    output_cursor.execute("SELECT stock, date FROM gaps WHERE gaptype = 'GAP_UP' order by stock")
    gap_up_stocks = output_cursor.fetchall()

    # Print unique stocks
    unique_stocks = set(stock for stock, _ in gap_up_stocks)
    print(f"Unique stocks for gap up: {len(unique_stocks)}")

    current_stock = None

    for stock, date in gap_up_stocks:
        if current_stock != stock:
            current_stock = stock
            print(f"Processing stock for gap up: {stock}")

        try:
            minute_db = _determine_minute_db(stock)
            minute_data = _fetch_minute_data(minute_db, stock, date)

            # Calculate the best sell and buy times
            best_sell_time, best_buy_time, max_profit, profit_percentage = _calculate_optimal_trade_times(
                minute_data, ('09:17:00', '09:25:00'), ('09:26:00', '15:15:00'), True)

            # Insert the result into the optimal trades table
            if best_sell_time and best_buy_time:
                    _insert_optimal_trade(minute_optimal_cursor, stock, 'GAP_UP', date, 'SELL', best_sell_time, 'BUY', best_buy_time, max_profit, profit_percentage)
        except Exception as e:
            print(f"Error processing stock {stock} on date {date}, Continuing to next item: {e}")
            continue

    # Commit the changes and close the connections
    minute_optimal_conn.commit()
    output_conn.close()
    minute_optimal_conn.close()


def find_optimum_trade_time_for_gapdown():
    """
    This function finds the optimum trade time for gap down stocks.
    """
    output_conn, output_cursor, minute_optimal_conn, minute_optimal_cursor = _setup_database_connections('day_wise_gaps.db', 'minute_wise_optimal.db')

    # Fetch GAP_DOWN stocks
    output_cursor.execute("SELECT stock, date FROM gaps WHERE gaptype = 'GAP_DOWN' order by stock")
    gap_down_stocks = output_cursor.fetchall()

    # Print unique stocks
    unique_stocks = set(stock for stock, _ in gap_down_stocks)
    print(f"Unique stocks for gap down: {len(unique_stocks)}")

    current_stock = None

    for stock, date in gap_down_stocks:
        if current_stock != stock:
            current_stock = stock
            print(f"Processing stock for gap down: {stock}")

        try:
            minute_db = _determine_minute_db(stock)
            minute_data = _fetch_minute_data(minute_db, stock, date)

            # Calculate the best buy and sell times
            best_buy_time, best_sell_time, max_profit, profit_percentage = _calculate_optimal_trade_times(
                minute_data, ('09:17:00', '09:25:00'), ('09:26:00', '15:15:00'), False)

            # Insert the result into the optimal trades table
            if best_buy_time and best_sell_time:
                _insert_optimal_trade(minute_optimal_cursor, stock, 'GAP_DOWN', date, 'BUY', best_buy_time, 'SELL', best_sell_time, max_profit, profit_percentage)
        except Exception as e:
            print(f"Error processing stock {stock} on date {date}, Continuing to next item: {e}")
            continue

    # Commit the changes and close the connections
    minute_optimal_conn.commit()
    output_conn.close()
    minute_optimal_conn.close()

def load_optimal_trades_into_dataframe():
    """
    This function loads the optimal_trades DB into a pandas dataframe.
    """
    output_conn, output_cursor, minute_optimal_conn, minute_optimal_cursor = _setup_database_connections('day_wise_gaps.db', 'minute_wise_optimal.db')
    
    # Fetch all data from the optimal_trades table
    minute_optimal_cursor.execute("SELECT * FROM optimal_trades")
    optimal_trades = minute_optimal_cursor.fetchall()
    
    # Convert the fetched data into a pandas DataFrame
    df = pd.DataFrame(optimal_trades, columns=[i[0] for i in minute_optimal_cursor.description])
    return df

def _get_db_file_path(db_name):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, '../../database/ohlc_data', db_name)

def _setup_database_connections(output_db_name, minute_optimal_db_name):
    output_db = _get_db_file_path(output_db_name)
    minute_optimal_db = _get_db_file_path(minute_optimal_db_name)

    # Create a connection to the output database
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()

    # Create a connection to the minute optimal database
    minute_optimal_conn = sqlite3.connect(minute_optimal_db)
    minute_optimal_cursor = minute_optimal_conn.cursor()

    # Create the output table if it doesn't exist
    minute_optimal_cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimal_trades (
            stock TEXT,
            gap_type TEXT,
            date DATETIME,
            entry_type TEXT,
            entry_time TIMESTAMP,
            exit_type TEXT,
            exit_time TIMESTAMP,
            profit REAL,
            profit_percentage REAL,
            UNIQUE(stock, date)
        )
    ''')

    return output_conn, output_cursor, minute_optimal_conn, minute_optimal_cursor


def _determine_minute_db(stock):
    first_char = stock[0].upper()
    if first_char.isdigit():
        return _get_db_file_path('0-9-minute.db')
    else:
        return _get_db_file_path(f'{first_char}-minute.db')


def _fetch_minute_data(minute_db, stock, date):
    # Connect to the minute database
    minute_conn = sqlite3.connect(minute_db)
    minute_cursor = minute_conn.cursor()

    # Properly quote the table name to handle special characters
    query = f'SELECT ts, close FROM "{stock}" WHERE date(ts) = ? ORDER BY ts;'
    minute_cursor.execute(query, (date,))
    minute_data = minute_cursor.fetchall()

    minute_conn.close()
    return minute_data


def _calculate_optimal_trade_times(minute_data, entry_window, exit_window, is_gap_up):
    best_entry_time = None
    best_exit_time = None
    max_profit = float('-inf')
    profit_percentage = 0

    # Find the best entry time
    entry_candidates = [(ts, close) for ts, close in minute_data if entry_window[0] <= ts[11:19] <= entry_window[1]]

    # Find the best exit time
    exit_candidates = [(ts, close) for ts, close in minute_data if exit_window[0] <= ts[11:19] <= exit_window[1]]

    # Calculate the best entry and exit times
    for entry_time, entry_price in entry_candidates:
        for exit_time, exit_price in exit_candidates:
            if exit_time > entry_time:
                if is_gap_up:
                    # For GAP UP, entry is sell and exit is buy
                    profit = entry_price - exit_price
                else:
                    # For GAP DOWN, entry is buy and exit is sell
                    profit = exit_price - entry_price

                if profit > max_profit:
                    max_profit = profit
                    profit_percentage = (profit / entry_price) * 100
                    best_entry_time = entry_time
                    best_exit_time = exit_time

    return best_entry_time, best_exit_time, max_profit, profit_percentage


def _insert_optimal_trade(minute_optimal_cursor, stock, gap_type, date, entry_type, entry_time, exit_type, exit_time, profit, profit_percentage):
    minute_optimal_cursor.execute(
        'INSERT OR IGNORE INTO optimal_trades (stock, gap_type, date, entry_type, entry_time, exit_type, exit_time, profit, profit_percentage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (stock, gap_type, date, entry_type, entry_time, exit_type, exit_time, profit, profit_percentage)
    )


# Execute the function if the script is run as the main module
if __name__ == "__main__":
    # Uncomment the following lines to generate day_wise_gaps.db and minute_wise_optimal.db again
    # find_and_store_daily_gaps()
    # find_optimum_trade_time_for_gapup()
    # find_optimum_trade_time_for_gapdown()
    
    # Convert the optimal_trades DB into a pandas dataframe
    df = load_optimal_trades_into_dataframe()
    print(df.head())

    # TODO: Further analysis
