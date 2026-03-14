import pandas as pd
import numpy as np
import ta
import os

def load_data(data_dir):
    """Loads the SP500 index data and the constituents data."""
    stocks_path = os.path.join(data_dir, 'all_stocks_5yr.csv')
    index_path = os.path.join(data_dir, 'HistoricalData.csv')
    
    if not os.path.exists(stocks_path) or not os.path.exists(index_path):
        print(f"Error: Data files not found in {data_dir}. Please make sure 'all_stocks_5yr.csv' and 'HistoricalData.csv' are present.")
        return None, None
        
    stocks = pd.read_csv(stocks_path, parse_dates=['Date' if 'Date' in pd.read_csv(stocks_path, nrows=0).columns else 'date'])
    sp500 = pd.read_csv(index_path, parse_dates=['Date' if 'Date' in pd.read_csv(index_path, nrows=0).columns else 'date'])
    
    # Standardize column names stripping spaces
    stocks.columns = [str(col).lower().strip() for col in stocks.columns]
    sp500.columns = [str(col).lower().strip() for col in sp500.columns]
    
    if 'name' in stocks.columns:
        stocks = stocks.rename(columns={'name': 'ticker'})
        
    return stocks, sp500

def compute_features(df):
    """
    Computes Bollinger Bands, RSI, and MACD for each ticker.
    """
    # Sort by ticker and date chronologically
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    def apply_ta(group):
        if len(group) < 30:
            return group
        
        close = group['close']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        group['bb_bbm'] = bb.bollinger_mavg()
        group['bb_bbh'] = bb.bollinger_hband()
        group['bb_bbl'] = bb.bollinger_lband()
        group['bb_pos'] = (close - group['bb_bbl']) / (group['bb_bbh'] - group['bb_bbl'] + 1e-8)
        
        # RSI
        group['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=close)
        group['macd'] = macd.macd()
        group['macd_signal'] = macd.macd_signal()
        group['macd_diff'] = macd.macd_diff()
        
        return group

    # Apply indicators per ticker
    df = df.groupby('ticker', group_keys=True).apply(apply_ta)
    
    # In Pandas 3.0+, apply might exclude the grouping column or put it in index
    if 'ticker' not in df.columns:
        if 'ticker' in df.index.names:
            df = df.reset_index(level='ticker')
        else:
            df = df.reset_index()
            
    return df

def build_dataset(data_dir='data'):
    stocks, sp500 = load_data(data_dir)
    if stocks is None:
        return
        
    print("Computing technical features...")
    df = compute_features(stocks)
    
    print("Constructing targets without leakage...")
    # Make sure it's sorted
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    # Calculate daily return: return(D, D+1) 
    df['return_1d'] = df.groupby('ticker')['close'].pct_change()
    
    # Target is return(D+1, D+2), which means shifting the return back by 2 days
    # (Since on day D, return_1d is return(D-1, D). So to get return(D+1, D+2), we shift return_1d back 2 days)
    df['target_return'] = df.groupby('ticker')['return_1d'].shift(-2)
    
    # Target: sign of the target return
    # 1 if positive return, -1 if negative, we can keep 0 for no change
    df['target'] = np.sign(df['target_return'])
    df['target'] = df['target'].replace(0, 1) # Prefer long if neutral, or you can drop 0s.
    
    # Drop NaNs that appear due to shifting and rolling windows
    df = df.dropna().reset_index(drop=True)
    
    # Set multi-index
    df = df.set_index(['date', 'ticker'])
    
    # Select our final columns
    features = ['bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_pos', 'rsi', 'macd', 'macd_signal', 'macd_diff']
    # Keep target_return to calculate strategy returns later
    cols = features + ['target', 'target_return']
    final_df = df[cols]
    
    print("Splitting train and test sets...")
    train = final_df[final_df.index.get_level_values('date') < '2017-01-01']
    test = final_df[final_df.index.get_level_values('date') >= '2017-01-01']
    
    # Save
    train.to_csv(os.path.join(data_dir, 'train.csv'))
    test.to_csv(os.path.join(data_dir, 'test.csv'))
    
    # Save SP500 for backtesting index comparison
    sp500.to_csv(os.path.join(data_dir, 'sp500_processed.csv'), index=False)
    
    print(f"Train subset shape: {train.shape}")
    print(f"Test subset shape: {test.shape}")
    print("Data processing complete!")

if __name__ == '__main__':
    # Ensure current working directory is the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_directory = os.path.join(project_root, 'data')
    
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
    
    build_dataset(data_directory)
