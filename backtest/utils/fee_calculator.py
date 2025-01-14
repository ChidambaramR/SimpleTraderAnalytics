def calculate_brokerage(order_value):
    """
    Calculate brokerage fee (0.3% or Rs 20, whichever is lower)
    """
    brokerage = min(order_value * 0.003, 20)
    return brokerage

def calculate_fees(entry_value, exit_value, trade_direction):
    """
    Calculate total transaction fees including brokerage, STT, and other charges.
    
    Args:
        entry_value: Total value at entry (quantity * price)
        exit_value: Total value at exit (quantity * price)
        trade_direction: 'LONG' or 'SHORT'
    
    Returns:
        Dictionary containing breakdown of fees and total fees
    """
    # Brokerage (both entry and exit)
    entry_brokerage = calculate_brokerage(entry_value)
    exit_brokerage = calculate_brokerage(exit_value)
    total_brokerage = entry_brokerage + exit_brokerage
    
    # STT (only on sell side)
    if trade_direction == 'LONG':
        # For long trades, STT on exit
        stt = exit_value * 0.00025
    else:
        # For short trades, STT on entry
        stt = entry_value * 0.00025
    
    # Transaction charges (both sides)
    entry_transaction = entry_value * 0.0000297
    exit_transaction = exit_value * 0.0000297
    total_transaction = entry_transaction + exit_transaction
    
    # SEBI charges (both sides)
    # Rs 10 per crore = 10/10000000 = 0.0000001 * value
    entry_sebi = entry_value * 0.0000001
    exit_sebi = exit_value * 0.0000001
    total_sebi = entry_sebi + exit_sebi
    
    # Stamp duty (only on buy side)
    if trade_direction == 'LONG':
        # For long trades, stamp duty on entry
        stamp_duty = min(entry_value * 0.00003, entry_value * 300 / 10000000)
    else:
        # For short trades, stamp duty on exit
        stamp_duty = min(exit_value * 0.00003, exit_value * 300 / 10000000)
    
    # GST (18% on brokerage + transaction charges + SEBI charges)
    gst = (total_brokerage + total_transaction + total_sebi) * 0.18
    
    # Calculate total fees
    total_fees = (total_brokerage + stt + total_transaction + 
                 total_sebi + stamp_duty + gst)
    
    return {
        'brokerage': total_brokerage,
        'stt': stt,
        'transaction_charges': total_transaction,
        'sebi_charges': total_sebi,
        'stamp_duty': stamp_duty,
        'gst': gst,
        'total': total_fees
    } 