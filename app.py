from flask import Flask, render_template, request
from database.gap_queries.queries import (
    analyze_daily_total_gaps,
    analyze_daily_gap_closures,
    analyze_daily_gap_ranges,
    analyze_daily_successful_gap_ranges,
    analyze_daily_gap_range_success_rates,
    analyze_first_minute_moves,
    analyze_first_minute_rest_of_day_moves
)
from backtest.router import run_backtest
from prod_stats.utils import get_in_market_ticks_data, get_pre_market_ticks_data, prepare_depth_data, get_trade_points, get_stock_logs
from trader_stats.utils import get_opening_gaps_trader_stats
from datetime import datetime

app = Flask(__name__)

def format_indian_currency(amount):
    """
    Formats a number in Indian currency style (with commas).
    Example: 2878980.82 -> 28,78,980.82
    """
    try:
        # Split the number into integer and decimal parts
        str_amount = f"{float(amount):.2f}"
        integer_part, decimal_part = str_amount.split('.')
        
        # Handle negative numbers
        is_negative = integer_part.startswith('-')
        if is_negative:
            integer_part = integer_part[1:]
        
        # Format integer part with Indian style grouping
        length = len(integer_part)
        if length <= 3:
            result = integer_part
        else:
            # First comma after 3 digits from right
            result = integer_part[-3:]
            # Then comma after every 2 digits
            remaining = integer_part[:-3]
            while remaining:
                result = remaining[-2:] + "," + result if len(remaining) >= 2 else remaining + "," + result
                remaining = remaining[:-2]
        
        # Add decimal part and handle negative sign
        formatted = f"{'-' if is_negative else ''}{result}.{decimal_part}"
        return formatted
    except:
        return str(amount)

# Register the custom filter
app.jinja_env.filters['indian_currency'] = format_indian_currency

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze/gaps', methods=['GET', 'POST'])
def analyze_gaps():
    results = None
    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        interval = request.form.get('interval')
        
        if interval == 'day':
            # Get daily analyses
            distribution_data = analyze_daily_total_gaps(from_date, to_date)
            closure_data = analyze_daily_gap_closures(from_date, to_date)
            ranges_data = analyze_daily_gap_ranges(from_date, to_date)
            successful_ranges_data = analyze_daily_successful_gap_ranges(from_date, to_date)
            success_rates_data = analyze_daily_gap_range_success_rates(from_date, to_date)
        else:
            # Placeholder for minute interval
            # TODO: Implement minute interval analysis
            return {'error': 'Minute interval analysis not yet implemented'}
        
        # Combine the results
        results = {
            'distribution': distribution_data,
            'closure': closure_data,
            'ranges': ranges_data,
            'successful_ranges': successful_ranges_data,
            'success_rates': success_rates_data
        }
        
    return render_template('analyze/gaps/index.html', results=results)

@app.route('/strategies/gaps/trading-gaps-daywise/without-sl-tp')
def gaps_without_sl_tp():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    force_run = request.args.get('force_run', 'false').lower() == 'true'
    
    if from_date and to_date:
        results = run_backtest('gaps_without_sl_tp', from_date, to_date, force_run=force_run)
        return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp.html', 
                             results=results)
    
    return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp.html')

@app.route('/strategies/gaps/trading-gaps-daywise/without-sl-tp-fixed-position')
def gaps_without_sl_tp_fixed_position():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    force_run = request.args.get('force_run', 'false').lower() == 'true'
    
    if from_date and to_date:
        results = run_backtest('gaps_without_sl_tp_fixed_position', from_date, to_date, force_run=force_run)
        return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp_fixed_position.html', 
                             results=results)
    
    return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp_fixed_position.html')

@app.route('/strategies/gaps/trading_gaps_first_minute/with_sl_tp')
def gaps_first_minute_with_sl_tp():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    force_run = request.args.get('force_run', 'false').lower() == 'true'
    
    if from_date and to_date:
        args = {
            'stop_loss_pct': float(request.args.get('stop_loss', 0.75)),
            'take_profit_pct': float(request.args.get('take_profit', 2)),
            'exit_time': request.args.get('exit_time', '09:16')
        }

        results = run_backtest('gaps_trading_first_minute_with_sl_tp', 
                             from_date, 
                             to_date, 
                             force_run=force_run,
                             args=args)
        return render_template('strategies/gaps/trading_gaps_first_minute/with_sl_tp.html', 
                             results=results)
    
    return render_template('strategies/gaps/trading_gaps_first_minute/with_sl_tp.html')

@app.route('/strategies/gaps/trading_gaps_leg2/leg2_sl_tp')
def gaps_leg2_sl_tp():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    force_run = request.args.get('force_run', 'false').lower() == 'true'
    
    if from_date and to_date:
        args = {
            'stop_loss_pct': float(request.args.get('stop_loss', 0.75)),
            'take_profit_pct': float(request.args.get('take_profit', 2)),
            'entry_time': request.args.get('entry_time', '09:17')
        }

        results = run_backtest('gaps_trading_sl_tp_leg2', 
                             from_date, 
                             to_date, 
                             force_run=force_run,
                             args=args)
        return render_template('strategies/gaps/trading_gaps_leg2/leg2_sl_tp.html', 
                             results=results)
    
    return render_template('strategies/gaps/trading_gaps_leg2/leg2_sl_tp.html')


@app.route('/analyze/gaps/first-minute', methods=['GET', 'POST'])
def analyze_gaps_first_minute():
    results = None
    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        analysis_time = request.form.get('analysis_time', '09:15')
        
        first_min_results = analyze_first_minute_moves(from_date, to_date, analysis_time)
        #rest_of_day_results = analyze_first_minute_rest_of_day_moves(from_date, to_date)
        
        # Combine both results
        results = {
            'total_instances': first_min_results['total_instances'],
            'details': first_min_results['details'],
            #'rest_of_day': rest_of_day_results
        }
        
    return render_template('analyze/gaps/first_minute.html', results=results)

@app.route('/analyze/gaps/further-moves', methods=['GET', 'POST'])
def analyze_moves_on_further_gaps():
    results = None
    error = None
    scenario_data = None
    
    if request.method == 'POST':
        try:
            from_date = request.form.get('from_date')
            to_date = request.form.get('to_date')
            analysis_minute = request.form.get('analysis_minute', '09:15')
            force_rerun = request.form.get('force_rerun') == 'on'
            
            if not from_date or not to_date:
                error = 'Missing required dates'
            else:
                print(f"Analyzing gaps with params: from={from_date}, to={to_date}, minute={analysis_minute}")
                
                # Call the analysis function with the specified minute
                analysis_results = analyze_first_minute_rest_of_day_moves(from_date, to_date, analysis_minute)
                
                if 'error' in analysis_results:
                    print(f"Analysis error: {analysis_results['error']}")
                    error = analysis_results['error']
                    results = None
                else:
                    print(f"Analysis completed successfully")
                    # Get all results except scenarios
                    results = {k: v for k, v in analysis_results.items() if k != 'scenarios'}
                    
                    # Generate and save chart images
                    if 'scenarios' in analysis_results:
                        import os
                        import matplotlib.pyplot as plt
                        import matplotlib.dates as mdates
                        from datetime import datetime
                        import random
                        
                        # Clear existing charts
                        import shutil
                        charts_dir = os.path.join('static', 'scenario_charts')
                        if os.path.exists(charts_dir):
                            shutil.rmtree(charts_dir)
                        os.makedirs(charts_dir)
                        
                        scenario_counts = {}
                        scenario_details = {}
                        
                        # Process each scenario type
                        for scenario_type, scenarios in analysis_results['scenarios'].items():
                            if not scenarios:  # Skip if no scenarios for this type
                                continue
                                
                            scenario_dir = os.path.join(charts_dir, scenario_type)
                            os.makedirs(scenario_dir)
                            
                            # Get all scenarios for this type
                            scenario_counts[scenario_type] = len(scenarios)
                            scenario_details[scenario_type] = []
                            
                            # Generate chart for each scenario
                            for i, scenario in enumerate(scenarios[:50]):  # Limit to 50 scenarios per type
                                # Create figure and axis with wider dimensions
                                plt.rcParams['figure.figsize'] = [20, 10]  # Make the figure wider
                                plt.rcParams['figure.dpi'] = 100
                                fig, ax = plt.subplots()
                                
                                # Convert time strings to datetime for proper x-axis
                                times = [datetime.strptime(t, '%H:%M') for t in [d['time'] for d in scenario['ohlc_data']]]
                                
                                # Plot candlesticks
                                for j, candle in enumerate(scenario['ohlc_data']):
                                    time = times[j]
                                    open_price = float(candle['open'])
                                    close = float(candle['close'])
                                    high = float(candle['high'])
                                    low = float(candle['low'])
                                    
                                    # Determine color based on price movement
                                    color = 'g' if close >= open_price else 'r'
                                    
                                    # Plot candle body
                                    ax.plot([time, time], [open_price, close], color=color, linewidth=3)
                                    # Plot wicks
                                    ax.plot([time, time], [low, high], color=color, linewidth=1)
                                
                                # Plot horizontal lines for previous close and day open
                                ax.axhline(y=scenario['prev_close'], color='red', linestyle='--', label='Previous Close')
                                ax.axhline(y=scenario['day_open'], color='blue', linestyle='--', label='Day Open')
                                
                                # Add first minute high/low line based on scenario type
                                first_min_data = scenario['ohlc_data'][0]  # First minute's data
                                if scenario_type.startswith('gap_up'):
                                    # For gap up scenarios, show first minute low
                                    first_min_low = float(first_min_data['low'])
                                    ax.axhline(y=first_min_low, color='orange', linestyle='--', label='First Min Low')
                                else:
                                    # For gap down scenarios, show first minute high
                                    first_min_high = float(first_min_data['high'])
                                    ax.axhline(y=first_min_high, color='orange', linestyle='--', label='First Min High')
                                
                                # Format x-axis
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                                plt.xticks(rotation=45)
                                
                                # Add grid
                                ax.grid(True, linestyle='--', alpha=0.7)
                                
                                # Add legend
                                ax.legend()
                                
                                # Add title
                                plt.title(f"{scenario['stock']} - {scenario['date']} (Gap: {scenario['gap_percent']:.2f}%)")
                                
                                # Adjust layout to prevent text cutoff
                                plt.tight_layout()
                                
                                # Save chart with extra padding
                                chart_filename = f"chart_{i}.png"
                                chart_path = os.path.join(scenario_dir, chart_filename)
                                plt.savefig(chart_path, bbox_inches='tight', dpi=100, pad_inches=0.5)
                                plt.close()
                                
                                # Store scenario details
                                scenario_details[scenario_type].append({
                                    'index': i,
                                    'image_url': f"/static/scenario_charts/{scenario_type}/chart_{i}.png"
                                })
                        
                        # Store scenario data for template
                        scenario_data = {
                            'counts': scenario_counts,
                            'details': scenario_details,
                            'current_scenario': 'gap_up_crossed'
                        }
            
        except Exception as e:
            print(f"Error in analyze_moves_on_further_gaps: {str(e)}")
            import traceback
            traceback.print_exc()
            error = str(e)
            results = None
        
    return render_template('analyze/gaps/analyze_moves_on_further_gaps.html', 
                         results=results,
                         scenario_data=scenario_data,
                         error=error)

@app.route('/trader-stats/opening-gaps-trader')
def opening_gaps_trader_stats():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    result_type = request.args.get('result_type', 'ANY')
    
    if from_date and to_date:
        try:
            # Convert dates to datetime
            from_date = datetime.strptime(from_date, '%Y-%m-%d')
            to_date = datetime.strptime(to_date, '%Y-%m-%d')
            
            # Get stats data
            df, stats = get_opening_gaps_trader_stats(from_date, to_date, result_type)
            
            return render_template('trader_stats/opening_gaps_trader.html', 
                                 df=df,
                                 stats=stats,
                                 from_date=from_date.strftime('%Y-%m-%d'),
                                 to_date=to_date.strftime('%Y-%m-%d'),
                                 result_type=result_type)
        except Exception as e:
            print(f"Error processing stats: {str(e)}")
            return render_template('trader_stats/opening_gaps_trader.html', error=str(e))
    
    return render_template('trader_stats/opening_gaps_trader.html')

@app.route('/analyze/pre-market-ticks', methods=['GET', 'POST'])
def analyze_pre_market_ticks():
    df = None
    summary = None
    
    # Get date and symbol from either POST or GET
    date = request.form.get('date') or request.args.get('date')
    symbol = request.form.get('symbol') or request.args.get('symbol')
    
    # If we have parameters, treat it as a form submission
    if date and symbol:
        df, summary = get_pre_market_ticks_data(date, symbol)
            
    return render_template('analyze/pre_market_ticks.html', 
                          df=df, 
                          summary=summary,
                          date=date,
                          symbol=symbol,
                          prepare_depth_data=prepare_depth_data)

@app.route('/analyze/in-market-ticks', methods=['GET', 'POST'])
def analyze_in_market_ticks():
    df = None
    summary = None
    trades = None
    logs = []  # Initialize logs list
    
    # Get date and symbol from either POST or GET
    date = request.form.get('date') or request.args.get('date')
    symbol = request.form.get('symbol') or request.args.get('symbol')
    
    # If we have parameters, treat it as a form submission
    if date and symbol:
        df, summary = get_in_market_ticks_data(date, symbol)
        trades = get_trade_points(date, symbol)
        
        if df is not None:
            # Get logs for this stock
            logs = get_stock_logs(date, symbol)
            print(f"Found {len(logs)} logs for {symbol}")  # Debug print
            print("Sample logs:", logs[:2])  # Print first two logs if any
            
        return render_template('analyze/in_market_ticks.html', 
                             date=date, 
                             symbol=symbol, 
                             df=df, 
                             summary=summary,
                             trades=trades,
                             logs=logs)
    
    return render_template('analyze/in_market_ticks.html', date=date, symbol=symbol)

if __name__ == '__main__':
    app.run(debug=True)
