import boto3
import json
import os
from datetime import datetime
import logging

def sync_data_from_s3():
    """
    Syncs data from S3 buckets based on manifest comparison.
    Downloads new/modified files and maintains directory structure.
    """
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Load AWS credentials
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3'
        )
        
        # Load manifest
        manifest_path = os.path.join(current_dir, 'manifest.json')
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        
        # Initialize manifest sections if they don't exist
        if 'ledger' not in manifest:
            manifest['ledger'] = {}
        if 'logs' not in manifest:
            manifest['logs'] = {}
        
        # Sync Ledger files
        bucket = 'simpletrader-working-bucket-ajith'
        ledger_prefix = 'SimpleTraderLedger/'
        
        # Handle pagination for ledger files
        paginator = s3_client.get_paginator('list_objects_v2')
        ledger_pages = paginator.paginate(
            Bucket=bucket,
            Prefix=ledger_prefix
        )
        
        # Process ledger files
        for page in ledger_pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_ledger.csv'):
                    # Extract date from filename
                    filename = os.path.basename(key)
                    date_str = filename.split('_')[0]
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # Create directory structure
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')
                    
                    local_dir = os.path.join(current_dir, 'data', year, month, day)
                    local_path = os.path.join(local_dir, 'ledger.csv')
                    
                    # Check if file needs to be downloaded
                    etag = obj['ETag'].strip('"')
                    if key not in manifest['ledger'] or manifest['ledger'][key] != etag:
                        logger.info(f"Downloading ledger file: {key}")
                        os.makedirs(local_dir, exist_ok=True)
                        s3_client.download_file(bucket, key, local_path)
                        manifest['ledger'][key] = etag
        
        # Sync Logs files
        logs_prefix = 'SimpleTraderLogs/'
        
        # Handle pagination for logs files
        logs_pages = paginator.paginate(
            Bucket=bucket,
            Prefix=logs_prefix
        )
        
        # Process logs files
        for page in logs_pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = os.path.basename(key)
                
                # Check if file matches our pattern
                if filename.startswith('pre_market_ticks_data_') or filename.startswith('pre_market_data_') or filename.startswith('ticks_data_') or filename == "logs.txt":
                    # Extract date from parent directory
                    date_str = os.path.basename(os.path.dirname(key))
                    if not date_str:
                        continue
                        
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        continue
                    
                    # Create directory structure
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')
                    
                    local_dir = os.path.join(current_dir, 'data', year, month, day)
                    
                    if filename.startswith('pre_market'):
                    # Always save as pre_market_ticks_data_*.txt
                        stock_name = filename.split('_')[-1].replace('.txt', '')
                        local_filename = f'pre_market_ticks_data_{stock_name}.txt'
                    else:
                        local_filename = filename

                    local_path = os.path.join(local_dir, local_filename)
                    
                    # Check if file needs to be downloaded
                    etag = obj['ETag'].strip('"')
                    if key not in manifest['logs'] or manifest['logs'][key] != etag:
                        logger.info(f"Downloading log file: {key}")
                        os.makedirs(local_dir, exist_ok=True)
                        s3_client.download_file(bucket, key, local_path)
                        manifest['logs'][key] = etag
        
        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info("S3 sync completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error syncing data from S3: {str(e)}")
        return False

sync_data_from_s3()