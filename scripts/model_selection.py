import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from gridsearch import BlockingTimeSeriesSplit

def analyze_model(project_root):
    data_path = os.path.join(project_root, 'data', 'train.csv')
    model_path = os.path.join(project_root, 'results', 'selected-model', 'selected_model.pkl')
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Data or selected model not found.")
        return
        
    print("Loading data and model for analysis...")
    df = pd.read_csv(data_path, parse_dates=['date']).set_index(['date', 'ticker']).sort_index()
    
    X = df.drop(columns=['target', 'target_return'])
    y = df['target']
    features_list = X.columns.tolist()
    
    model = joblib.load(model_path)
    cv = BlockingTimeSeriesSplit(n_splits=10)
    
    # To store metrics
    metrics = []
    feature_importances = []
    
    print("Evaluating model across folds...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit model on training fold
        model.fit(X_train, y_train)
        
        # Predictions
        train_probs = model.predict_proba(X_train)[:, 1]
        train_preds = model.predict(X_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)
        
        # Metrics
        y_train_mapped = (y_train == 1).astype(int) # Map back to 0/1 for metrics
        y_val_mapped = (y_val == 1).astype(int)
        
        train_auc = roc_auc_score(y_train_mapped, train_probs)
        train_acc = accuracy_score(y_train, train_preds)
        train_ll = log_loss(y_train_mapped, train_probs)
        
        val_auc = roc_auc_score(y_val_mapped, val_probs)
        val_acc = accuracy_score(y_val, val_preds)
        val_ll = log_loss(y_val_mapped, val_probs)
        
        metrics.extend([
            {'Fold': fold, 'Dataset': 'Train', 'AUC': train_auc, 'Accuracy': train_acc, 'LogLoss': train_ll},
            {'Fold': fold, 'Dataset': 'Validation', 'AUC': val_auc, 'Accuracy': val_acc, 'LogLoss': val_ll}
        ])
        
        # Feature Importance (Extract from RandomForest if it's the classifier)
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            # Sort and get top 10
            indices = np.argsort(importances)[::-1][:10]
            for rank, idx in enumerate(indices):
                feature_importances.append({
                    'Fold': fold,
                    'Rank': rank + 1,
                    'Feature': features_list[idx],
                    'Importance': importances[idx]
                })

    cv_dir = os.path.join(project_root, 'results', 'cross-validation')
    os.makedirs(cv_dir, exist_ok=True)
    
    # Process Metrics
    metrics_df = pd.DataFrame(metrics).set_index(['Fold', 'Dataset'])
    metrics_df.to_csv(os.path.join(cv_dir, 'ml_metrics_train.csv'))
    print("\nMetrics summary saved to ml_metrics_train.csv")
    
    # Process Feature Importances
    fi_df = pd.DataFrame(feature_importances)
    # Pivot to match formatting
    if not fi_df.empty:
         fi_df.to_csv(os.path.join(cv_dir, 'top_10_feature_importance.csv'), index=False)
         print("Feature importances saved.")
    
    # Plot AUC
    plot_df = metrics_df.reset_index()
    train_aucs = plot_df[plot_df['Dataset'] == 'Train']['AUC'].values
    val_aucs = plot_df[plot_df['Dataset'] == 'Validation']['AUC'].values
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(10), train_aucs, marker='o', label='Train AUC')
    plt.plot(range(10), val_aucs, marker='o', label='Validation AUC')
    plt.title('AUC across Cross-Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'metric_train.png'), dpi=300)
    plt.close()
    
    print("AUC plot saved to metric_train.png")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    analyze_model(project_root)
