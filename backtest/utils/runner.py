def run_backtest_with_cache(strategy_name, from_date, to_date, backtest_func, args={}):
    """
    Generic backtest runner with caching functionality.
    
    Args:
        strategy_name: Unique identifier for the strategy
        from_date: Start date for backtest
        to_date: End date for backtest
        backtest_func: Function that implements the backtest
    """
    try:        
        return backtest_func(from_date, to_date, args) # Run backtest
    except Exception as e:
        return {'error': f"Error in backtest runner: {str(e)}"}