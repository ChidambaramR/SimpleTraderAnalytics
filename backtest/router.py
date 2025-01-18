from .gaps.trading_gaps_daywise.without_sl_tp import run_backtest as run_gaps_without_sl_tp
from .gaps.trading_gaps_daywise.without_sl_tp_fixed_position import run_backtest_fixed_position
from .gaps.trading_gaps_first_minute.with_sl_tp import run_gaps_first_minute_with_sl_tp
def run_backtest(strategy_name, from_date, to_date, force_run=False, args={}):
    """
    Routes backtest requests to appropriate strategy implementations.
    
    Args:
        strategy_name: Identifier for the strategy to run
        from_date: Start date for backtest
        to_date: End date for backtest
        force_run: If True, bypasses cache
        
    Returns:
        Dictionary containing backtest results
    """
    strategy_map = {
        'gaps_without_sl_tp': run_gaps_without_sl_tp,
        'gaps_without_sl_tp_fixed_position': run_backtest_fixed_position,
        'gaps_trading_first_minute_with_sl_tp': run_gaps_first_minute_with_sl_tp
    }
    
    if strategy_name not in strategy_map:
        return {
            'error': f'Unknown strategy: {strategy_name}',
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
    
    return strategy_map[strategy_name](from_date, to_date, force_run, args) 