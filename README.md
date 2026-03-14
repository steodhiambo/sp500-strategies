# SP500 Machine Learning Trading Strategy

This project builds a quantitative trading strategy that uses an ML model to outperform the S&P 500 index. The model predicts the sign of future stock returns using technical indicators (Bollinger Bands, RSI, MACD) specifically structured to prevent data leakage.

## Prerequisites
1. Install requirements into your environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the required dataset files inside the `data/` directory:
   - `HistoricalData.csv` (SP500 index data)
   - `all_stocks_5yr.csv` (SP500 constituent data)

## Pipeline Execution
Run the scripts in the following sequential order from the root project directory:

1. **Feature Engineering**
   Processes the raw data, computes technical indicators, shifts target variables based on $return(D+1, D+2)$, drops NaNs, and splits into train/test sets split at the year 2017.
   ```bash
   python scripts/features_engineering.py
   ```

2. **Setup Cross-Validation and Grid Search**
   Runs a custom `BlockingTimeSeriesSplit` (to ensure out-of-sample moving window evaluation with minimum 2-years train history) and performs `GridSearchCV` on a RandomForest classifer to find best hyperparameters.
   ```bash
   python scripts/gridsearch.py
   ```

3. **Model Selection & Metric Evaluation**
   Analyzes the model selected by GridSearch across the 10 folds. Computes AUC, Accuracy, LogLoss, and outputs Top 10 feature importances to `results/cross-validation/`.
   ```bash
   python scripts/model_selection.py
   ```

4. **Signal Generation**
   Uses Out-of-Fold (OOF) predictions to create daily probability investment signals on the validation sets, concatenating them into `ml_signal.csv`.
   ```bash
   python scripts/create_signal.py
   ```

5. **Strategy Backtesting**
   Converts the generated ML signal into a long-short financial strategy ($1 invested per day, equally weighted into highest and lowest signal stocks). Calculates PnL using the forward returns and evaluates maximum drawdown against the S&P 500 buy-and-hold benchmarks.
   ```bash
   python scripts/strategy.py
   ```

## Expected Results
- Check the `results/` folder for generated output.
- Analysis outputs include `blocking_time_series_split.png`, `metric_train.png`, and a formatted quantitative `report.md`.
