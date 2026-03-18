import os
import pandas as pd
import numpy as np
import joblib
from gridsearch import BlockingTimeSeriesSplit
from sklearn.base import clone

def create_signal(project_root):
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    model_path = os.path.join(project_root, 'results', 'selected-model', 'selected_model.pkl')
    
    if not os.path.exists(train_path) or not os.path.exists(model_path) or not os.path.exists(test_path):
        print("Required files not found. Ensure train.csv, test.csv and selected_model.pkl exist.")
        return
        
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, parse_dates=['date']).set_index(['date', 'ticker']).sort_index()
    test_df = pd.read_csv(test_path, parse_dates=['date']).set_index(['date', 'ticker']).sort_index()
    
    X_train = train_df.drop(columns=['target', 'target_return'])
    y_train = train_df['target']
    
    X_test = test_df.drop(columns=['target', 'target_return'])
    # y_test is not needed for signal generation, but we keep the index
    
    # Load previously found parameters
    model = joblib.load(model_path)
    cv = BlockingTimeSeriesSplit(n_splits=10)
    
    oof_predictions = []
    
    print("Generating OOF machine learning signals for the train set...")
    for train_idx, val_idx in cv.split(X_train):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        
        # Clone avoids keeping previously fitted state
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        
        probs = fold_model.predict_proba(X_fold_val)[:, 1]
        
        val_dates = X_train.index.get_level_values('date')[val_idx]
        val_tickers = X_train.index.get_level_values('ticker')[val_idx]
        
        fold_preds = pd.DataFrame({
            'date': val_dates,
            'ticker': val_tickers,
            'signal': probs
        })
        oof_predictions.append(fold_preds)
        
    print("Generating signal for the test set (trained on full train data)...")
    final_model = clone(model)
    final_model.fit(X_train, y_train)
    test_probs = final_model.predict_proba(X_test)[:, 1]
    
    test_preds = pd.DataFrame({
        'date': X_test.index.get_level_values('date'),
        'ticker': X_test.index.get_level_values('ticker'),
        'signal': test_probs
    })
    
    print("Concatenating train and test signals...")
    all_signals = pd.concat(oof_predictions + [test_preds], axis=0)
    
    # Needs a double index ordered with the probability
    all_signals = all_signals.set_index(['date', 'ticker']).sort_index()
    
    out_dir = os.path.join(project_root, 'results', 'selected-model')
    os.makedirs(out_dir, exist_ok=True)
    all_signals.to_csv(os.path.join(out_dir, 'ml_signal.csv'))
    
    print(f"Signal generated for {len(all_signals)} samples and saved to results/selected-model/ml_signal.csv")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_signal(project_root)
