from flask import Flask, render_template, request
from database.gap_queries.queries import (
    analyze_daily_total_gaps,
    analyze_daily_gap_closures,
    analyze_daily_gap_ranges,
    analyze_daily_successful_gap_ranges,
    analyze_daily_gap_range_success_rates
)
from backtest.gaps.trading_gaps_daywise.without_sl_tp import run_backtest

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

@app.route('/facts/gaps', methods=['GET', 'POST'])
def gaps():
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
        
    return render_template('facts/gaps/index.html', results=results)

@app.route('/strategies/gaps/trading-gaps-daywise/without-sl-tp')
def gaps_without_sl_tp():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    force_run = request.args.get('force_run', 'false').lower() == 'true'
    
    if from_date and to_date:
        results = run_backtest(from_date, to_date, force_run=force_run)
        return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp.html', 
                             results=results)
    
    # Default view without results
    return render_template('strategies/gaps/trading_gaps_daywise/without_sl_tp.html')

if __name__ == '__main__':
    app.run(debug=True)
