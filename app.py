from flask import Flask, render_template, request
from database.gap_queries.queries import total_gaps

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
        
        # Using the imported query_gaps function
        results = total_gaps(from_date, to_date, interval)
        
    return render_template('facts/gaps/index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
