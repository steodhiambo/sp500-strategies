import os
import pandas as pd
import matplotlib.pyplot as plt

def run_strategy(project_root):
    signal_path = os.path.join(project_root, 'results', 'selected-model', 'ml_signal.csv')
    train_data_path = os.path.join(project_root, 'data', 'train.csv')
    test_data_path = os.path.join(project_root, 'data', 'test.csv')
    sp500_path = os.path.join(project_root, 'data', 'sp500_processed.csv')
    
    if not os.path.exists(signal_path) or not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Required files not found.")
        return
        
    print("Loading signal and market data...")
    signal_df = pd.read_csv(signal_path, parse_dates=['date']).set_index(['date', 'ticker'])
    
    # Combined target returns for evaluation
    train_df = pd.read_csv(train_data_path, parse_dates=['date']).set_index(['date', 'ticker'])
    test_df = pd.read_csv(test_data_path, parse_dates=['date']).set_index(['date', 'ticker'])
    combined_targets = pd.concat([train_df[['target_return']], test_df[['target_return']]], axis=0)
    
    strategy_data = signal_df.join(combined_targets, how='inner')
    
    def backtest_strategy(group):
        group = group.sort_values(by='signal', ascending=False)
        k = 10 
        if len(group) < 2 * k:
            k = len(group) // 2
            if k == 0:
                group['pnl'] = 0
                return group
        
        group['position'] = 0
        group.iloc[:k, group.columns.get_loc('position')] = 1
        group.iloc[-k:, group.columns.get_loc('position')] = -1
        
        weight = 1.0 / (2 * k)
        group['pnl'] = group['position'] * group['target_return'] * weight
        return group
    
    print("Backtesting strategy...")
    strategy_data = strategy_data.groupby('date', group_keys=False).apply(backtest_strategy)
    daily_pnl = strategy_data.groupby('date')['pnl'].sum()
    
    # SP500 processing
    sp500 = pd.read_csv(sp500_path, parse_dates=['date']).sort_values(by='date').set_index('date')
    sp500['return_1d'] = sp500['close'].pct_change()
    sp500['target_return'] = sp500['return_1d'].shift(-2)
    sp500_daily_pnl = sp500['target_return']
    
    aligned_data = pd.DataFrame({'Strategy': daily_pnl, 'SP500': sp500_daily_pnl}).dropna()
    aligned_data['Strategy_CumPnL'] = aligned_data['Strategy'].cumsum()
    aligned_data['SP500_CumPnL'] = aligned_data['SP500'].cumsum()
    
    # Metrics calculation
    def max_drawdown(return_series):
        cum_returns = (1 + return_series).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
    
    split_date = pd.Timestamp('2017-01-01')
    train_mask = aligned_data.index < split_date
    test_mask = aligned_data.index >= split_date
    
    train_results = aligned_data[train_mask]
    test_results = aligned_data[test_mask]
    
    def get_metrics(df):
        if len(df) == 0: return 0, 0, 0, 0
        strat_pnl = df['Strategy'].sum()
        sp500_pnl = df['SP500'].sum()
        strat_mdd = max_drawdown(df['Strategy'])
        sp500_mdd = max_drawdown(df['SP500'])
        return strat_pnl, sp500_pnl, strat_mdd, sp500_mdd

    tr_s_pnl, tr_idx_pnl, tr_s_mdd, tr_idx_mdd = get_metrics(train_results)
    ts_s_pnl, ts_idx_pnl, ts_s_mdd, ts_idx_mdd = get_metrics(test_results)
    
    print(f"\n--- Train Results (< 2017) ---")
    print(f"Strategy PnL: {tr_s_pnl:.4f}, MDD: {tr_s_mdd:.2%}")
    print(f"SP500 PnL: {tr_idx_pnl:.4f}, MDD: {tr_idx_mdd:.2%}")
    
    print(f"\n--- Test Results (>= 2017) ---")
    print(f"Strategy PnL: {ts_s_pnl:.4f}, MDD: {ts_s_mdd:.2%}")
    print(f"SP500 PnL: {ts_idx_pnl:.4f}, MDD: {ts_idx_mdd:.2%}")
    
    out_dir = os.path.join(project_root, 'results', 'strategy')
    os.makedirs(out_dir, exist_ok=True)
    aligned_data.to_csv(os.path.join(out_dir, 'results.csv'))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(aligned_data.index, aligned_data['Strategy_CumPnL'], label='Strategy PnL', color='blue')
    ax1.plot(aligned_data.index, aligned_data['SP500_CumPnL'], label='SP500 PnL', color='orange')
    
    # Add separation line
    if split_date in aligned_data.index or (aligned_data.index.min() < split_date < aligned_data.index.max()):
        ax1.axvline(x=split_date, color='red', linestyle='--', label='Train/Test Split (2017)')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL')
    ax1.set_title('Strategy vs SP500 Cumulative PnL (Train & Test)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'strategy.png'), dpi=300)
    plt.close()
    
    report_content = f"""# Strategy Backtesting Report

## Methodology

### Features Used
- Bollinger Bands (Mvg Avg, High/Low Bands, relative Position)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD, Signal, Diff)

### Machine Learning Pipeline
- **Imputer**: SimpleImputer (median)
- **Scaler**: StandardScaler
- **Classifier**: RandomForestClassifier

### Cross-Validation
- **Method**: Blocking Time Series Split
- **Folds**: 10
- **Train Length**: Minimum 2 years.

### Strategy Chosen
- **Description**: Stock picking (Long 10 best, Short 10 worst) per day.
- **Investment**: $1 daily total.

## Results

### Train Set (< 2017)
- **Strategy PnL**: {tr_s_pnl:.4f}
- **SP500 PnL**: {tr_idx_pnl:.4f}
- **Strategy Max Drawdown**: {tr_s_mdd:.2%}
- **SP500 Max Drawdown**: {tr_idx_mdd:.2%}

### Test Set (>= 2017)
- **Strategy PnL**: {ts_s_pnl:.4f}
- **SP500 PnL**: {ts_idx_pnl:.4f}
- **Strategy Max Drawdown**: {ts_s_mdd:.2%}
- **SP500 Max Drawdown**: {ts_idx_mdd:.2%}

![PnL Plot](strategy.png)
"""
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write(report_content)
    print("\nBacktesting complete. Report and plot updated.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    run_strategy(project_root)
