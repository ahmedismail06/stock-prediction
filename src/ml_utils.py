import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-labels
    Pipeline: Train -> Gap (Purge) -> Test -> Gap (Embargo)
    """
    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if isinstance(X, pd.DataFrame):
            indices = np.arange(X.shape[0])
        else:
            indices = X
            
        for train_indices, test_indices in super(PurgedKFold, self).split(indices):
            
            # 1. Basic difference (standard KFold does this, but good to be explicit)
            train_indices = np.setdiff1d(train_indices, test_indices)
            
            # 2. Define Gaps
            # Gap 1 (Embargo): Immediately after test set. 
            # Gap 2 (Purge): Immediately before test set (overlapping labels).
            
            max_test_idx = test_indices.max()
            min_test_idx = test_indices.min()
            
            # Calculate embargo size (e.g., 1% of data)
            embargo_span = int(X.shape[0] * self.pct_embargo)
            
            # Buffer for purging (look-ahead bias into test start)
            # Since target is 1m return, we need at least 1 step buffer
            purge_span = 1 
            
            # Mask for "Too close before test"
            # If train < min_test, it must be < (min_test - purge_span)
            mask_before = (train_indices < min_test_idx) & (train_indices >= (min_test_idx - purge_span))
            
            # Mask for "Too close after test"
            # If train > max_test, it must be > (max_test + embargo_span)
            mask_after = (train_indices > max_test_idx) & (train_indices <= (max_test_idx + embargo_span))
            
            # Combine masks (indices to REMOVE)
            mask_remove = mask_before | mask_after
            
            # Keep indices that are NOT in the remove mask
            train_indices = train_indices[~mask_remove]
            
            yield train_indices, test_indices

def apply_cross_validation(data, target_col='target_1m'):
    """
    Helper function to run the CV and print shapes.
    """
    # Create the validator
    cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
    
    print(f"Starting Cross-Validation on data shape: {data.shape}")
    
    fold = 0
    for train_ix, test_ix in cv.split(data):
        fold += 1
        
        train_dates = data.iloc[train_ix].index.get_level_values('date')
        test_dates = data.iloc[test_ix].index.get_level_values('date')
        
        print(f"\nFold {fold}:")
        print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} ({len(train_ix)} samples)")
        print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_ix)} samples)")