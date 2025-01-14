from .db_utils import save_backtest_results, get_backtest_results

def run_backtest_with_cache(strategy_name, from_date, to_date, backtest_func, force_run=False):
    """
    Generic backtest runner with caching functionality.
    
    Args:
        strategy_name: Unique identifier for the strategy
        from_date: Start date for backtest
        to_date: End date for backtest
        backtest_func: Function that implements the backtest
        force_run: If True, runs backtest regardless of cache
    """
    try:
        if not force_run:
            # Try to get cached results
            cached_results = get_backtest_results(strategy_name, from_date, to_date)
            if cached_results:
                return cached_results
        
        # Run backtest
        results = backtest_func(from_date, to_date)
        
        # Only cache if there's no error
        if 'error' not in results:
            save_backtest_results(strategy_name, from_date, to_date, results)
        
        return results
        
    except Exception as e:
        # Return a dictionary with all expected fields
        return {
            'error': f"Error in backtest runner: {str(e)}",
            'total_trades': 0,
            'win_ratio': 0,
            'total_invested': 0,
            'profit_1x': 0,
            'profit_5x': 0,
            'max_drawdown_1x': 0,
            'max_drawdown_5x': 0,
            'roi_1x': 0,
            'roi_5x': 0
        } 