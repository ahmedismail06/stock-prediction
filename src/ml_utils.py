import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

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


# Stress testing by injecting Gaussian noise
def stress_test_model(model, X, y, noise_levels=[0.1, 0.5, 1.0, 2.0]):
    """
    Evaluates model performance under increasing levels of noise.
    
    Parameters:
    -----------
    model : trained sklearn model
    X : features (pd.DataFrame)
    y : targets (pd.Series)
    noise_levels : list of floats (standard deviations of noise)
    
    Returns:
    --------
    pd.DataFrame : Results of stress test
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score

    results = []
    
    # Base performance (No Noise)
    base_pred = model.predict(X)
    # We use correlation (IC) because that's your main metric
    base_ic = np.corrcoef(y, base_pred)[0, 1]
    
    results.append({
        'noise_level': 0.0,
        'ic': base_ic,
        'ic_change_pct': 0.0
    })
    
    print(f"Base IC (No Noise): {base_ic:.4f}")

    for noise in noise_levels:
        # Create random noise with the same shape as X
        # scale=noise determines how strong the noise is
        noise_matrix = np.random.normal(loc=0.0, scale=noise, size=X.shape)
        
        # Add noise to features
        X_stressed = X + noise_matrix
        
        # Predict
        pred_stressed = model.predict(X_stressed)
        
        # Measure Performance
        stressed_ic = np.corrcoef(y, pred_stressed)[0, 1]
        
        # Calculate how much performance dropped
        drop = (stressed_ic - base_ic) / base_ic
        
        print(f"Noise {noise}: IC = {stressed_ic:.4f} (Drop: {drop:.1%})")
        
        results.append({
            'noise_level': noise,
            'ic': stressed_ic,
            'ic_change_pct': drop
        })
        
    return pd.DataFrame(results)