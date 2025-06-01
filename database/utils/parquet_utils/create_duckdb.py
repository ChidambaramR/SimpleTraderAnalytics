import os
import duckdb

# Folder with parquet files
parquet_folder = '/Users/ajith/Desktop/Stocks/SimpleTraderAnalytics/database/ohlc_data/parquet'

# Output DuckDB file
output_db = '/Users/ajith/Desktop/Stocks/SimpleTraderAnalytics/database/ohlc_data/merged_parquet.duckdb'

# Connect to persistent DuckDB file
conn = duckdb.connect(output_db)

file_count = 0

for file_name in os.listdir(parquet_folder):
    if file_name.endswith('.parquet'):
        file_count = file_count + 1
        print(f"Processing file {file_count}: {file_name}")
        file_path = os.path.join(parquet_folder, file_name)
        table_name = os.path.splitext(file_name)[0]

        print(f"Adding {file_name} as table `{table_name}`")
        conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM \'{file_path}\'')
    
conn.close()

print(f"\nDuckDB created at: {output_db}")
