import os
import pandas as pd
import matplotlib.pyplot as plt

def run_strategy(project_root):
    signal_path = os.path.join(project_root, 'results', 'selected-model', 'ml_signal.csv')
    data_path = os.path.join(project_root, 'data', 'train.csv')
    sp500_path = os.path.join(project_root, 'data', 'sp500_processed.csv')
    test_data_path = os.path.join(project_root, 'data', 'test.csv')
    
    if not os.path.exists(signal_path) or not os.path.exists(data_path):
        print("Required files not found.")
        return
        
    print("Loading signal and market data...")
    # Signal index: date, ticker. Contains 'signal' (probability of going up)
    signal_df = pd.read_csv(signal_path, parse_dates=['date']).set_index(['date', 'ticker'])
    
    # We need the target_return from the original datasets to evaluate PnL
    train_df = pd.read_csv(data_path, parse_dates=['date']).set_index(['date', 'ticker'])
    
    # Test data might be used if signal was computed over it, but for OOF signal, it's only on train
    # If the user wants an out-of-sample generation, they'd use model_selection.py on the test set.
    # We'll calculate OOF strategy returns here.
    strategy_data = signal_df.join(train_df[['target_return']], how='inner')
    
    def backtest_strategy(group):
        group = group.sort_values(by='signal', ascending=False)
        
        k = 10 # top 10 long, bottom 10 short
        if len(group) < 2 * k:
            k = len(group) // 2
            if k == 0:
                group['pnl'] = 0
                return group
                
        group['position'] = 0
        group.iloc[:k, group.columns.get_loc('position')] = 1
        group.iloc[-k:, group.columns.get_loc('position')] = -1
        
        # $1 total per day -> sum of absolute weights is 1
        weight = 1.0 / (2 * k)
        group['pnl'] = group['position'] * group['target_return'] * weight
        
        return group
    
    print("Backtesting top/bottom K strategy...")
    strategy_data = strategy_data.groupby('date', group_keys=False).apply(backtest_strategy)
    
    daily_pnl = strategy_data.groupby('date')['pnl'].sum()
    
    # SP500 daily return (shift -2 to match the target_return definition: return D+1 to D+2)
    sp500 = pd.read_csv(sp500_path, parse_dates=['date']).sort_values(by='date').set_index('date')
    sp500['return_1d'] = sp500['close'].pct_change()
    sp500['target_return'] = sp500['return_1d'].shift(-2)
    sp500_daily_pnl = sp500['target_return']
    
    aligned_data = pd.DataFrame({'Strategy': daily_pnl, 'SP500': sp500_daily_pnl}).dropna()
    aligned_data['Strategy_CumPnL'] = aligned_data['Strategy'].cumsum()
    aligned_data['SP500_CumPnL'] = aligned_data['SP500'].cumsum()
    
    def max_drawdown(return_series):
        cum_returns = (1 + return_series).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
        
    strat_mdd = max_drawdown(aligned_data['Strategy'])
    sp500_mdd = max_drawdown(aligned_data['SP500'])
    
    print(f"Strategy Max Drawdown: {strat_mdd:.2%}")
    print(f"SP500 Max Drawdown: {sp500_mdd:.2%}")
    print(f"Strategy Final PnL: {aligned_data['Strategy_CumPnL'].iloc[-1]:.4f}")
    
    out_dir = os.path.join(project_root, 'results', 'strategy')
    os.makedirs(out_dir, exist_ok=True)
    
    aligned_data.to_csv(os.path.join(out_dir, 'results.csv'))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(aligned_data.index, aligned_data['Strategy_CumPnL'], label='Strategy PnL', color='blue')
    ax1.plot(aligned_data.index, aligned_data['SP500_CumPnL'], label='SP500 PnL', color='orange')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL')
    ax1.set_title('Strategy vs SP500 Cumulative PnL')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'strategy.png'), dpi=300)
    plt.close()
    
    print("Strategy evaluation saved.")
    
    report_content = f"""# Strategy Backtesting Report

## Methodology

### Features Used
- Bollinger Bands (Mvg Avg, High/Low Bands, relative Position)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD, Signal, Diff)

### Machine Learning Pipeline
- **Imputer**: SimpleImputer (median strategy)
- **Scaler**: StandardScaler
- **Classifier**: RandomForestClassifier

### Cross-Validation
- **Method**: Blocking Time Series Split (Sliding Window without overlap)
- **Folds**: 10
- **Train Length**: Minimum 2 years of history.

### Strategy Chosen
- **Description**: Stock picking (Long K best, Short K worst) based on highest and lowest output signal probabilities. K=10.
- **Investment**: Invests $1 total per day, equally weighted across the 2K positions ($0.05 per position).
- Leverages Out-of-Fold (OOF) daily predictions. The PnL calculation maps day D's signal with the actual return achieved between D+1 and D+2 to avoid forward-looking data leakage.

## Results
- **Final Strategy Cumulative PnL**: {aligned_data['Strategy_CumPnL'].iloc[-1]:.4f}
- **Final SP500 Cumulative PnL**: {aligned_data['SP500_CumPnL'].iloc[-1]:.4f}
- **Strategy Max Drawdown**: {strat_mdd:.2%}
- **SP500 Max Drawdown**: {sp500_mdd:.2%}

![PnL Plot](strategy.png)
"""
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write(report_content)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    run_strategy(project_root)
