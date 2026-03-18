# Strategy Backtesting Report

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
- **Strategy PnL**: -0.0363
- **SP500 PnL**: 0.1118
- **Strategy Max Drawdown**: -11.72%
- **SP500 Max Drawdown**: -14.16%

### Test Set (>= 2017)
- **Strategy PnL**: -0.1215
- **SP500 PnL**: 0.1702
- **Strategy Max Drawdown**: -12.36%
- **SP500 Max Drawdown**: -7.79%

![PnL Plot](strategy.png)
