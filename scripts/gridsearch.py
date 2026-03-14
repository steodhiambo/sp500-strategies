import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class BlockingTimeSeriesSplit:
    def __init__(self, n_splits=10):
        self.n_splits = n_splits
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        
    def split(self, X, y=None, groups=None):
        dates = pd.Series(X.index.get_level_values('date')).unique()
        dates = np.sort(dates)
        n_dates = len(dates)
        
        # Ensure > 2 years of history for train (approx 504 business days)
        min_train_size = 504 
        if n_dates < min_train_size + self.n_splits:
            min_train_size = n_dates // 2
            
        test_size = (n_dates - min_train_size) // self.n_splits
        
        date_col = X.index.get_level_values('date')
        
        for i in range(self.n_splits):
            train_start = i * test_size
            train_end = train_start + min_train_size
            test_start = train_end
            test_end = test_start + test_size
            
            if i == self.n_splits - 1:
                test_end = n_dates
                
            train_dates = set(dates[train_start:train_end])
            test_dates = set(dates[test_start:test_end])
            
            # Use boolean indexing for faster lookup
            train_mask = [d in train_dates for d in date_col]
            test_mask = [d in test_dates for d in date_col]
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx

def plot_cv_indices(cv, X, ax, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    # To reduce plotting time, we sample indices if the dataset is large
    step = max(1, len(X) // 1000)
    
    for ii, (tr, tt) in enumerate(cv.split(X)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        
        # Scatter plot for visualization
        ax.scatter(range(0, len(indices), step), [ii + .5] * len(indices[::step]),
                   c=indices[::step], marker='_', lw=lw, cmap=plt.cm.coolwarm,
                   vmin=-.2, vmax=1.2)

    ax.set(yticks=np.arange(cv.n_splits) + .5, yticklabels=[str(i) for i in range(1, cv.n_splits + 1)],
           ylabel="CV iteration", xlabel="Sample index",
           title="Blocking Time Series Split")
    ax.legend([Patch(color=plt.cm.coolwarm(.8)), Patch(color=plt.cm.coolwarm(.2))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
              
def run_gridsearch(project_root):
    data_path = os.path.join(project_root, 'data', 'train.csv')
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    print("Loading training data for grid search...")
    df = pd.read_csv(data_path, parse_dates=['date']).set_index(['date', 'ticker']).sort_index()
    
    # We drop 'target_return' as it's not a feature, and target is our y.
    X = df.drop(columns=['target', 'target_return'])
    y = df['target']
    
    # Setup CV
    cv = BlockingTimeSeriesSplit(n_splits=10)
    
    # Plot CV
    cv_dir = os.path.join(project_root, 'results', 'cross-validation')
    os.makedirs(cv_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    print("Plotting CV splits...")
    plot_cv_indices(cv, X, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'blocking_time_series_split.png'), dpi=300)
    plt.close()
    
    # Setup Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Parameter grid for Random Forest
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 10],
        'clf__min_samples_leaf': [10, 50]
    }
    
    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1, # Keep outer n_jobs 1 to avoid memory overload, RF is parallelized
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    # Save the selected model
    model_dir = os.path.join(project_root, 'results', 'selected-model')
    os.makedirs(model_dir, exist_ok=True)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(model_dir, 'selected_model.pkl'))
    
    with open(os.path.join(model_dir, 'selected_model.txt'), 'w') as f:
        f.write(f"Best Parameters:\n{grid_search.best_params_}\n")
        f.write(f"Best CV AUC: {grid_search.best_score_:.4f}\n")
        
    print("Grid search complete and model saved.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    run_gridsearch(project_root)
