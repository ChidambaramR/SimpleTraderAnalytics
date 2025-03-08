import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

STOCK = 'ACC'

# Create output directory path
OUTPUT_DIR = f'analysis/data_dump/{STOCK}'
# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _determine_minute_db(stock):
    first_char = stock[0].upper()
    if first_char.isdigit():
        return '0-9-minute.db'
    else:
        return f'{first_char}-minute.db'

# Path to the databases
GAPS_DB_PATH = '../database/ohlc_data/day_wise_gaps.db'
MINUTE_DB_PATH = f'../database/ohlc_data/{_determine_minute_db(STOCK)}'

# Function to load gap days for a specific stock
def load_gap_days(stock):
    script_dir = os.path.dirname(__file__)
    conn = sqlite3.connect(os.path.join(script_dir, GAPS_DB_PATH))
    query = f"SELECT date, gaptype, pctdiff FROM gaps WHERE stock='{stock}' ORDER BY date"
    gap_days = pd.read_sql_query(query, conn)
    conn.close()
    return gap_days

# Function to load minute data for a specific day
def load_minute_data(stock, date):
    script_dir = os.path.dirname(__file__)
    conn = sqlite3.connect(os.path.join(script_dir, MINUTE_DB_PATH))
    # Get data for the whole trading day (9:15 AM to 3:30 PM)
    query = f"SELECT * FROM {stock} WHERE ts LIKE '{date}%' ORDER BY ts"
    minute_data = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp to datetime
    minute_data['ts'] = pd.to_datetime(minute_data['ts'])
    
    return minute_data

# Function to extract intraday features for the ML model
def extract_intraday_features(minute_data, entry_window_start='09:17:00', entry_window_end='09:30:00'):
    # Filter data for the entry window (9:17 AM to 9:30 AM)
    entry_window_start_time = datetime.strptime(entry_window_start, '%H:%M:%S').time()
    entry_window_end_time = datetime.strptime(entry_window_end, '%H:%M:%S').time()
    
    entry_data = minute_data[
        (minute_data['ts'].dt.time >= entry_window_start_time) & 
        (minute_data['ts'].dt.time <= entry_window_end_time)
    ]
    
    if entry_data.empty:
        return None
    
    # Calculate features within the entry window
    features = {}
    
    # Opening price of the day (first candle)
    if not minute_data.empty:
        features['day_open'] = minute_data.iloc[0]['open']
        features['day_open_high'] = minute_data.iloc[0]['high']
        features['day_open_low'] = minute_data.iloc[0]['low']
        features['day_open_close'] = minute_data.iloc[0]['close']
        features['day_open_volume'] = minute_data.iloc[0]['volume']
    else:
        return None
    
    # Price action features from the entry window
    features['entry_window_max_high'] = entry_data['high'].max()
    features['entry_window_min_low'] = entry_data['low'].min()
    features['entry_window_open'] = entry_data.iloc[0]['open']
    features['entry_window_close'] = entry_data.iloc[-1]['close']
    features['entry_window_volume'] = entry_data['volume'].sum()
    
    # Calculate price movement within entry window
    features['entry_window_range'] = features['entry_window_max_high'] - features['entry_window_min_low']
    features['entry_window_movement'] = features['entry_window_close'] - features['entry_window_open']
    features['entry_window_volatility'] = features['entry_window_range'] / features['entry_window_open'] * 100
    
    # Candle patterns
    pattern_columns = []
    for idx, row in entry_data.iterrows():
        if pd.notna(row['CANDLE_PATTERN']) and row['CANDLE_PATTERN']:
            patterns = row['CANDLE_PATTERN'].split(',')
            pattern_columns.extend(patterns)
    
    # Unique patterns
    pattern_columns = list(set(pattern_columns))
    features['unique_patterns'] = len(pattern_columns)
    features['has_marubozu'] = 1 if any('MARUBOZU' in pattern for pattern in pattern_columns) else 0
    features['has_doji'] = 1 if any('DOJI' in pattern for pattern in pattern_columns) else 0
    features['has_engulfing'] = 1 if any('ENGULFING' in pattern for pattern in pattern_columns) else 0
    
    # Color distribution
    features['green_candles'] = len(entry_data[entry_data['CANDLE_COLOR'] == 'GREEN'])
    features['red_candles'] = len(entry_data[entry_data['CANDLE_COLOR'] == 'RED'])
    features['grey_candles'] = len(entry_data[entry_data['CANDLE_COLOR'] == 'GREY'])
    
    # Volume features
    features['volume_std'] = entry_data['volume'].std() if len(entry_data) > 1 else 0
    features['max_volume_candle'] = entry_data['volume'].max()
    features['min_volume_candle'] = entry_data['volume'].min()
    
    return features

# Function to simulate trades and calculate profit
def simulate_trades(minute_data, entry_time, exit_time, position_type, stop_loss_pct, take_profit_pct):
    # Ensure numeric parameters
    try:        
        # Convert parameters to float if needed
        if isinstance(stop_loss_pct, dict):
            print(f"WARNING: stop_loss_pct is a dictionary: {stop_loss_pct}")
            stop_loss_pct = 2.0  # Default value
        else:
            stop_loss_pct = float(stop_loss_pct)
            
        if isinstance(take_profit_pct, dict):
            print(f"WARNING: take_profit_pct is a dictionary: {take_profit_pct}")
            take_profit_pct = 3.0  # Default value
        else:
            take_profit_pct = float(take_profit_pct)
    except Exception as e:
        print(f"Error converting stop_loss_pct or take_profit_pct to float: {e}")
        stop_loss_pct = 2.0  # Default value
        take_profit_pct = 3.0  # Default value
    
    # Filter for entry and exit times
    entry_data = minute_data[minute_data['ts'].dt.time == datetime.strptime(entry_time, '%H:%M:%S').time()]
    
    if entry_data.empty:
        return 0  # No entry found
    
    entry_price = entry_data.iloc[0]['close']
    
    # Calculate stop loss and take profit levels
    if position_type == 'BUY':
        stop_loss = entry_price * (1 - stop_loss_pct / 100)
        take_profit = entry_price * (1 + take_profit_pct / 100)
    else:  # SELL
        stop_loss = entry_price * (1 + stop_loss_pct / 100)
        take_profit = entry_price * (1 - take_profit_pct / 100)
    
    # Find data after entry time
    post_entry_data = minute_data[minute_data['ts'] > entry_data.iloc[0]['ts']]
    
    # Set initial profit
    profit_pct = 0
    exit_price = None
    
    # Manually set exit time if specified
    target_exit_time = datetime.strptime(exit_time, '%H:%M:%S').time()
    
    # Loop through post-entry data to check for stop loss, take profit, or exit time
    for idx, row in post_entry_data.iterrows():
        current_price = row['close']
        current_time = row['ts'].time()
        
        # Check if stop loss hit
        if (position_type == 'BUY' and current_price <= stop_loss) or \
           (position_type == 'SELL' and current_price >= stop_loss):
            if position_type == 'BUY':
                profit_pct = (stop_loss - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - stop_loss) / entry_price * 100
            exit_price = stop_loss
            break
            
        # Check if take profit hit
        elif (position_type == 'BUY' and current_price >= take_profit) or \
             (position_type == 'SELL' and current_price <= take_profit):
            if position_type == 'BUY':
                profit_pct = (take_profit - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - take_profit) / entry_price * 100
            exit_price = take_profit
            break
            
        # Check if exit time reached
        elif current_time >= target_exit_time:
            if position_type == 'BUY':
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - current_price) / entry_price * 100
            exit_price = current_price
            break
    
    # If no exit conditions met by end of day, calculate profit based on last price
    if exit_price is None and not post_entry_data.empty:
        last_price = post_entry_data.iloc[-1]['close']
        if position_type == 'BUY':
            profit_pct = (last_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - last_price) / entry_price * 100
        exit_price = last_price
    
    return profit_pct

# Main function to prepare dataset for ML
def prepare_dataset(stock):
    # Check if cached dataset exists
    cache_path = os.path.join(OUTPUT_DIR, f"{stock}_dataset_cached.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"Error loading cached dataset: {e}. Will recompute.")
    
    print(f"Preparing new dataset for {stock}...")
    # Load gap days
    gap_days = load_gap_days(stock)
    
    # Prepare dataset
    dataset = []
    
    for idx, row in gap_days.iterrows():
        date = row['date']
        print(f"Processing {date}...")
        gap_type = row['gaptype']
        pct_diff = row['pctdiff']
        
        try:
            # Load minute data for the day
            minute_data = load_minute_data(stock, date)
            
            if minute_data.empty:
                continue
            
            # Extract features
            features = extract_intraday_features(minute_data)
            
            if features is None:
                continue
            
            # Add gap information to features
            features['gap_type'] = 1 if gap_type == 'GAP_UP' else 0
            features['gap_percent'] = pct_diff
            
            # CHANGE 1: Restrict entry time between 09:17 and 09:20
            entry_times = [f'09:{i:02d}:00' for i in range(17, 21)]  # Only 09:17 to 09:20
            
            # And different exit times throughout the day
            exit_times = [f'{h:02d}:{m:02d}:00' for h in range(9, 16) for m in range(0, 60, 15) if not (h == 9 and m < 30) and not (h == 15 and m > 30)]
            
            best_profit = -float('inf')
            best_strategy = None
            
            # CHANGE 2: Choose position type based on gap type
            if gap_type == 'GAP_UP':
                position_types = ['SELL']  # Only SELL for GAP_UP
            else:  # GAP_DOWN
                position_types = ['BUY']   # Only BUY for GAP_DOWN
            
            # CHANGE 3 & 4: Modify stop loss and take profit percentages
            stop_loss_percentages = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # From 1% to 5% in steps of 0.5%
            take_profit_percentages = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # From 1% to 4% in steps of 0.5%
            
            for entry_time in entry_times:
                for position_type in position_types:
                    for stop_loss_pct in stop_loss_percentages:
                        for take_profit_pct in take_profit_percentages:
                            for exit_time in exit_times:
                                # Skip invalid combinations (exit time before entry time)
                                entry_hour, entry_minute = map(int, entry_time.split(':')[:2])
                                exit_hour, exit_minute = map(int, exit_time.split(':')[:2])
                                
                                if exit_hour < entry_hour or (exit_hour == entry_hour and exit_minute <= entry_minute):
                                    continue
                                
                                profit = simulate_trades(
                                    minute_data, 
                                    entry_time, 
                                    exit_time, 
                                    position_type, 
                                    stop_loss_pct, 
                                    take_profit_pct
                                )
                                
                                if profit > best_profit:
                                    best_profit = profit
                                    best_strategy = {
                                        'entry_time': entry_time,
                                        'exit_time': exit_time,
                                        'position_type': position_type,
                                        'stop_loss_pct': stop_loss_pct,
                                        'take_profit_pct': take_profit_pct,
                                    }
            
            # Add the best strategy to the features
            if best_strategy is not None:
                features.update(best_strategy)
                features['profit'] = best_profit
                dataset.append(features)
                
        except Exception as e:
            print(f"Error processing {date}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Cache the DataFrame
    try:
        print(f"Saving dataset cache to {cache_path}")
        df.to_pickle(cache_path)
    except Exception as e:
        print(f"Error saving dataset cache: {e}")
    
    return df

# Function to train ML models
def train_ml_models(data):
    print(f"Starting model training with {len(data)} data points...")
    
    # Split features and targets
    X = data.drop(['entry_time', 'exit_time', 'position_type', 'stop_loss_pct', 'take_profit_pct', 'profit'], axis=1)
    
    # Create separate target variables
    print("Preparing target variables...")
    y_position = data['position_type'].map({'BUY': 1, 'SELL': 0})
    y_entry_hour = data['entry_time'].apply(lambda x: int(x.split(':')[0]))
    y_entry_minute = data['entry_time'].apply(lambda x: int(x.split(':')[1]))
    y_exit_hour = data['exit_time'].apply(lambda x: int(x.split(':')[0]))
    y_exit_minute = data['exit_time'].apply(lambda x: int(x.split(':')[1]))
    y_stop_loss = data['stop_loss_pct']
    y_take_profit = data['take_profit_pct']
    
    # Split data into train and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_position_train, y_position_test = train_test_split(X, y_position, test_size=0.2, random_state=42)
    _, _, y_entry_hour_train, y_entry_hour_test = train_test_split(X, y_entry_hour, test_size=0.2, random_state=42)
    _, _, y_entry_minute_train, y_entry_minute_test = train_test_split(X, y_entry_minute, test_size=0.2, random_state=42)
    _, _, y_exit_hour_train, y_exit_hour_test = train_test_split(X, y_exit_hour, test_size=0.2, random_state=42)
    _, _, y_exit_minute_train, y_exit_minute_test = train_test_split(X, y_exit_minute, test_size=0.2, random_state=42)
    _, _, y_stop_loss_train, y_stop_loss_test = train_test_split(X, y_stop_loss, test_size=0.2, random_state=42)
    _, _, y_take_profit_train, y_take_profit_test = train_test_split(X, y_take_profit, test_size=0.2, random_state=42)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train position type model (buy or sell)
    print("Training position type model...")
    position_model = RandomForestClassifier(n_estimators=100, random_state=42)
    position_model.fit(X_train_scaled, y_position_train)
    position_pred = position_model.predict(X_test_scaled)
    position_accuracy = accuracy_score(y_position_test, position_pred)
    print(f"Position model accuracy: {position_accuracy:.4f}")
    
    # Train entry time model (hour and minute)
    print("Training entry time models...")
    entry_hour_model = RandomForestRegressor(n_estimators=100, random_state=42)
    entry_hour_model.fit(X_train_scaled, y_entry_hour_train)
    entry_hour_pred = entry_hour_model.predict(X_test_scaled)
    entry_hour_mae = mean_absolute_error(y_entry_hour_test, entry_hour_pred)
    print(f"Entry hour MAE: {entry_hour_mae:.4f}")
    
    entry_minute_model = RandomForestRegressor(n_estimators=100, random_state=42)
    entry_minute_model.fit(X_train_scaled, y_entry_minute_train)
    entry_minute_pred = entry_minute_model.predict(X_test_scaled)
    entry_minute_mae = mean_absolute_error(y_entry_minute_test, entry_minute_pred)
    print(f"Entry minute MAE: {entry_minute_mae:.4f}")
    
    # Train exit time model (hour and minute)
    print("Training exit time models...")
    exit_hour_model = RandomForestRegressor(n_estimators=100, random_state=42)
    exit_hour_model.fit(X_train_scaled, y_exit_hour_train)
    exit_hour_pred = exit_hour_model.predict(X_test_scaled)
    exit_hour_mae = mean_absolute_error(y_exit_hour_test, exit_hour_pred)
    print(f"Exit hour MAE: {exit_hour_mae:.4f}")
    
    exit_minute_model = RandomForestRegressor(n_estimators=100, random_state=42)
    exit_minute_model.fit(X_train_scaled, y_exit_minute_train)
    exit_minute_pred = exit_minute_model.predict(X_test_scaled)
    exit_minute_mae = mean_absolute_error(y_exit_minute_test, exit_minute_pred)
    print(f"Exit minute MAE: {exit_minute_mae:.4f}")
    
    # Train stop loss and take profit models
    print("Training stop loss and take profit models...")
    stop_loss_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stop_loss_model.fit(X_train_scaled, y_stop_loss_train)
    stop_loss_pred = stop_loss_model.predict(X_test_scaled)
    stop_loss_mae = mean_absolute_error(y_stop_loss_test, stop_loss_pred)
    print(f"Stop loss MAE: {stop_loss_mae:.4f}")
    
    take_profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
    take_profit_model.fit(X_train_scaled, y_take_profit_train)
    take_profit_pred = take_profit_model.predict(X_test_scaled)
    take_profit_mae = mean_absolute_error(y_take_profit_test, take_profit_pred)
    print(f"Take profit MAE: {take_profit_mae:.4f}")
    
    # Save models
    print("Saving models to disk...")
    model_prefix = os.path.join(OUTPUT_DIR, f"{STOCK}")
    joblib.dump(position_model, f'{model_prefix}_position_model.pkl')
    joblib.dump(entry_hour_model, f'{model_prefix}_entry_hour_model.pkl')
    joblib.dump(entry_minute_model, f'{model_prefix}_entry_minute_model.pkl')
    joblib.dump(exit_hour_model, f'{model_prefix}_exit_hour_model.pkl')
    joblib.dump(exit_minute_model, f'{model_prefix}_exit_minute_model.pkl')
    joblib.dump(stop_loss_model, f'{model_prefix}_stop_loss_model.pkl')
    joblib.dump(take_profit_model, f'{model_prefix}_take_profit_model.pkl')
    joblib.dump(scaler, f'{model_prefix}_feature_scaler.pkl')
    
    print("Model training completed successfully!")
    return {
        'position_model': position_model,
        'entry_hour_model': entry_hour_model,
        'entry_minute_model': entry_minute_model,
        'exit_hour_model': exit_hour_model,
        'exit_minute_model': exit_minute_model,
        'stop_loss_model': stop_loss_model,
        'take_profit_model': take_profit_model,
        'scaler': scaler
    }

# Function to make predictions for new gap days
def predict_trading_strategy(models, features):
    # Scale features
    try:
        # Convert features dictionary to a pandas DataFrame to ensure proper column ordering
        features_df = pd.DataFrame([features])
        
        # Get the feature names from the scaler
        scaler_feature_names = models['scaler'].feature_names_in_ if hasattr(models['scaler'], 'feature_names_in_') else None
        if scaler_feature_names is not None:
            # Ensure correct column order
            features_df = features_df[scaler_feature_names]
        
        # Now transform using the correctly formatted DataFrame
        scaled_features = models['scaler'].transform(features_df)
    except Exception as e:
        print(f"Error scaling features: {e}")
        import traceback
        traceback.print_exc()
        # Return a default prediction rather than exiting
        return {
            'position_type': 'BUY',
            'position_confidence': 0.5,
            'entry_time': '09:17:00',
            'exit_time': '15:15:00',
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0
        }
    
    # Predict position type
    position_prob = models['position_model'].predict_proba(scaled_features)[0]
    position_type = 'BUY' if position_prob[1] > 0.5 else 'SELL'
    confidence = float(max(position_prob))  # Ensure it's a float
    
    # Predict entry time
    entry_hour = max(9, min(9, round(models['entry_hour_model'].predict(scaled_features)[0])))
    entry_minute = max(17, min(30, round(models['entry_minute_model'].predict(scaled_features)[0])))
    entry_time = f"{entry_hour:02d}:{entry_minute:02d}:00"
    
    # Predict exit time
    exit_hour = max(entry_hour, min(15, round(models['exit_hour_model'].predict(scaled_features)[0])))
    exit_minute = round(models['exit_minute_model'].predict(scaled_features)[0])
    # Ensure exit time is after entry time
    if exit_hour == entry_hour and exit_minute <= entry_minute:
        exit_minute = min(59, entry_minute + 15)
    exit_time = f"{exit_hour:02d}:{exit_minute:02d}:00"
    
    # Predict stop loss and take profit
    try:
        stop_loss_pct = float(max(0.5, min(5.0, models['stop_loss_model'].predict(scaled_features)[0])))
    except:
        stop_loss_pct = 2.0  # Default if prediction fails
        
    try:
        take_profit_pct = float(max(0.5, min(5.0, models['take_profit_model'].predict(scaled_features)[0])))
    except:
        take_profit_pct = 3.0  # Default if prediction fails
    
    return {
        'position_type': position_type,
        'position_confidence': confidence,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct
    }

# Function to evaluate model on historical data
def evaluate_model(stock, models):
    print(f"Starting model evaluation for {stock}...")
    
    # Load gap days
    gap_days = load_gap_days(stock)
    print(f"Loaded {len(gap_days)} gap days for evaluation")
    
    results = []
    processed_count = 0
    
    for idx, row in gap_days.iterrows():
        date = row['date']
        gap_type = row['gaptype']
        pct_diff = row['pctdiff']
        
        try:
            # Load minute data for the day
            minute_data = load_minute_data(stock, date)
            
            if minute_data.empty:
                print(f"No minute data found for {date}, skipping")
                continue
            
            # Extract features
            features = extract_intraday_features(minute_data)
            
            if features is None:
                print(f"Could not extract features for {date}, skipping")
                continue
            
            # Add gap information to features
            features['gap_type'] = 1 if gap_type == 'GAP_UP' else 0
            features['gap_percent'] = pct_diff
            
            # Get model predictions
            try:
                prediction = predict_trading_strategy(models, features)
                
                # Debug output
                print(f"Prediction for {date}: {prediction}")
                
                # Simulate trade with predicted strategy
                profit = simulate_trades(
                    minute_data,
                    prediction['entry_time'],
                    prediction['exit_time'],
                    prediction['position_type'],
                    prediction['stop_loss_pct'],
                    prediction['take_profit_pct']
                )
                
                # Ensure profit is a float
                if not isinstance(profit, (int, float)):
                    print(f"WARNING: profit is not numeric: {profit}")
                    profit = 0.0
                
                results.append({
                    'date': date,
                    'gap_type': gap_type,
                    'gap_pct': pct_diff,
                    'position_type': prediction['position_type'],
                    'entry_time': prediction['entry_time'],
                    'exit_time': prediction['exit_time'],
                    'stop_loss_pct': prediction['stop_loss_pct'],
                    'take_profit_pct': prediction['take_profit_pct'],
                    'profit_pct': profit
                })
                
            except Exception as inner_e:
                print(f"Error in prediction or simulation for {date}: {inner_e}")
                continue
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(gap_days)} days")
            
        except Exception as e:
            print(f"Error evaluating {date}: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(f"Evaluation complete. Generated results for {len(results_df)} trading days")
    
    # Handle the case where no valid results were generated
    if results_df.empty:
        print("WARNING: No valid trading days found for evaluation!")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['date', 'gap_type', 'gap_pct', 'position_type', 
                                    'entry_time', 'exit_time', 'stop_loss_pct', 
                                    'take_profit_pct', 'profit_pct'])
    
    # Calculate performance metrics
    total_trades = len(results_df)
    profitable_trades = len(results_df[results_df['profit_pct'] > 0])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    avg_profit = results_df['profit_pct'].mean()
    max_profit = results_df['profit_pct'].max()
    max_loss = results_df['profit_pct'].min()
    
    print(f"\nPerformance Summary for {stock}:")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average profit: {avg_profit:.2f}%")
    print(f"Maximum profit: {max_profit:.2f}%")
    print(f"Maximum loss: {max_loss:.2f}%")
    
    return results_df

# Function to run the entire pipeline
def run_pipeline(stock):
    print(f"Starting analysis for {stock}...")
    
    # Prepare dataset
    print("Preparing dataset...")
    data = prepare_dataset(stock)
    
    # Save dataset for later analysis
    data.to_csv(os.path.join(OUTPUT_DIR, f"{stock}_dataset.csv"), index=False)
    
    # Train models
    print("Training models...")
    models = train_ml_models(data)
    
    # Evaluate models
    print("Evaluating models...")
    evaluation = evaluate_model(stock, models)
    
    # Save evaluation results
    evaluation.to_csv(os.path.join(OUTPUT_DIR, f"{stock}_evaluation.csv"), index=False)
    
    print("Analysis complete!")
    return data, models, evaluation

# Execute the pipeline
if __name__ == "__main__":
    data, models, evaluation = run_pipeline(STOCK)
    
    # Serialize the results to files
    print("Serializing results to files...")
    
    # Data is already saved to CSV in run_pipeline
    print(f"Dataset already saved to {os.path.join(OUTPUT_DIR, f'{STOCK}_dataset.csv')}")
    
    # Models are already saved in train_ml_models using joblib
    print("Models already saved as PKL files")
    
    # Save evaluation results (already done in run_pipeline)
    print(f"Evaluation results already saved to {os.path.join(OUTPUT_DIR, f'{STOCK}_evaluation.csv')}")
    
    # Save combined results as JSON for easy access
    # Create a summary dictionary
    summary = {
        'stock': STOCK,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points': len(data),
        'evaluation_metrics': {
            'total_trades': len(evaluation),
            'win_rate': 0,
            'average_profit': 0,
            'max_profit': 0,
            'max_loss': 0
        }
    }
    
    # Only calculate metrics if we have valid trades
    if not evaluation.empty:
        summary['evaluation_metrics'] = {
            'total_trades': len(evaluation),
            'win_rate': (len(evaluation[evaluation['profit_pct'] > 0]) / len(evaluation)),
            'average_profit': float(evaluation['profit_pct'].mean()),
            'max_profit': float(evaluation['profit_pct'].max()),
            'max_loss': float(evaluation['profit_pct'].min())
        }
    
    # Save summary to JSON
    summary_path = os.path.join(OUTPUT_DIR, f"{STOCK}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to {summary_path}")
    print("All results successfully serialized!")