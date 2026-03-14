import os
import pandas as pd
import numpy as np
import joblib
from gridsearch import BlockingTimeSeriesSplit
from sklearn.base import clone

def create_signal(project_root):
    data_path = os.path.join(project_root, 'data', 'train.csv')
    model_path = os.path.join(project_root, 'results', 'selected-model', 'selected_model.pkl')
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Required files not found. Ensure train.csv and selected_model.pkl exist.")
        return
        
    print("Generating OOF machine learning signal...")
    df = pd.read_csv(data_path, parse_dates=['date']).set_index(['date', 'ticker']).sort_index()
    X = df.drop(columns=['target', 'target_return'])
    y = df['target']
    
    # Load previously found parameters
    model = joblib.load(model_path)
    cv = BlockingTimeSeriesSplit(n_splits=10)
    
    oof_predictions = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # Clone avoids keeping previously fitted state
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        # We need the probability of class 1
        probs = fold_model.predict_proba(X_val)[:, 1]
        
        val_dates = X.index.get_level_values('date')[val_idx]
        val_tickers = X.index.get_level_values('ticker')[val_idx]
        
        fold_preds = pd.DataFrame({
            'date': val_dates,
            'ticker': val_tickers,
            'signal': probs
        })
        oof_predictions.append(fold_preds)
        
    print("Concatenating signals...")
    signal_df = pd.concat(oof_predictions, axis=0)
    
    # Needs a double index ordered with the probability
    signal_df = signal_df.set_index(['date', 'ticker']).sort_values(by='signal', ascending=False)
    
    out_dir = os.path.join(project_root, 'results', 'selected-model')
    os.makedirs(out_dir, exist_ok=True)
    signal_df.to_csv(os.path.join(out_dir, 'ml_signal.csv'))
    
    print("Signal generated and saved to results/selected-model/ml_signal.csv")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_signal(project_root)
