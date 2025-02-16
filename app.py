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
from prod_stats.utils import get_pre_market_ticks_data
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

@app.route('/analyze/gaps/first-minute', methods=['GET', 'POST'])
def analyze_gaps_first_minute():
    results = None
    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        analysis_time = request.form.get('analysis_time', '09:15')
        
        first_min_results = analyze_first_minute_moves(from_date, to_date, analysis_time)
        rest_of_day_results = analyze_first_minute_rest_of_day_moves(from_date, to_date)
        
        # Combine both results
        results = {
            'total_instances': first_min_results['total_instances'],
            'details': first_min_results['details'],
            'rest_of_day': rest_of_day_results
        }
        
    return render_template('analyze/gaps/first_minute.html', results=results)

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
            df = get_opening_gaps_trader_stats(from_date, to_date, result_type)
            
            return render_template('trader_stats/opening_gaps_trader.html', 
                                 df=df,
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
                          symbol=symbol)

if __name__ == '__main__':
    app.run(debug=True)
