import os
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sqlite_folder = '/Users/ajith/Desktop/Stocks/SimpleTraderAnalytics/database/ohlc_data'
output_folder = 'database/ohlc_data/parquet'

os.makedirs(output_folder, exist_ok=True)

for db_file in os.listdir(sqlite_folder):
    if db_file.endswith('.db') and not db_file.endswith('day.db'):
        db_path = os.path.join(sqlite_folder, db_file)
        db_name = os.path.splitext(db_file)[0]

        print(f"Processing DB: {db_name}")
        con = sqlite3.connect(db_path)
        cursor = con.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            try:
                print(f"  Reading table: {table_name}")
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
                parquet_file = os.path.join(output_folder, f"{table_name}.parquet")

                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_file, compression='snappy')
                print(f"    > Saved to: {parquet_file}")
            except Exception as e:
                print(f"    > Error processing table {table_name}: {e}")

        con.close()
