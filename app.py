from flask import Flask, render_template, request
from database.gap_queries.queries import total_gaps, analyze_gap_closures, analyze_gap_ranges, analyze_successful_gap_ranges, analyze_gap_range_success_rates

app = Flask(__name__)

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
        
        # Get all analyses
        distribution_data = total_gaps(from_date, to_date, interval)
        closure_data = analyze_gap_closures(from_date, to_date, interval)
        ranges_data = analyze_gap_ranges(from_date, to_date, interval)
        successful_ranges_data = analyze_successful_gap_ranges(from_date, to_date, interval)
        success_rates_data = analyze_gap_range_success_rates(from_date, to_date, interval)
        
        # Combine the results
        results = {
            'distribution': distribution_data,
            'closure': closure_data,
            'ranges': ranges_data,
            'successful_ranges': successful_ranges_data,
            'success_rates': success_rates_data
        }
        
    return render_template('facts/gaps/index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
