import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from ml_utils import PurgedKFold
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

def train_ridge_baseline(dummy_data):
    """
    Train Ridge Regression baseline model with cross-validation.
    
    Parameters:
    -----------
    dummy_data : pd.DataFrame
        Engineered feature data with targets
        
    Returns:
    --------
    tuple : (mean_ic, ic_scores, all_predictions)
    """
    # --- 1. SETUP ---
    X = dummy_data.drop(columns=['target_1m', 'target_2m', 'target_3m', 'target_6m', 'target_12m'])
    y = dummy_data['target_1m']

    # Initialize the Validator
    cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

    # Initialize Storage for Results
    ic_scores = []
    predictions = []

    # --- 2. TRAINING LOOP ---
    logger.info(f"Training Ridge Regression on {X.shape[1]} features...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(dummy_data), 1):
        
        # A. Split Data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # B. Scale Features (Crucial for Ridge!)
        # We fit the scaler ONLY on training data to avoid leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Impute missing values with the mean
        imputer = SimpleImputer(strategy='mean')
        X_train_scaled = pd.DataFrame(
            imputer.fit_transform(X_train_scaled),
            columns=[f"feature_{i}" for i in range(X_train_scaled.shape[1])]
        )

        # Check if X_test_scaled exists and impute if necessary
        if 'X_test_scaled' in locals():
            X_test_scaled = pd.DataFrame(
                imputer.transform(X_test_scaled),
                columns=[f"feature_{i}" for i in range(X_test_scaled.shape[1])]
            )
        
        # C. Train Model (Ridge with default alpha=1.0)
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # D. Predict
        y_pred = model.predict(X_test_scaled)
        
        # E. Evaluate (Information Coefficient)
        # Spearman correlation checks if the RANKING is correct
        # (i.e. Did the model correctly predict AAPL > MSFT?)
        ic, p_val = spearmanr(y_test, y_pred)
        ic_scores.append(ic)
        
        logger.info(f"Fold {fold} IC: {ic:.4f}")
        
        # Save predictions for later analysis
        fold_preds = pd.DataFrame({'actual': y_test, 'predicted': y_pred}, index=y_test.index)
        predictions.append(fold_preds)

    # --- 3. SUMMARY ---
    mean_ic = np.mean(ic_scores)
    logger.info(f"\nAverage IC: {mean_ic:.4f}")
    logger.info(f"Standard Deviation of IC: {np.std(ic_scores):.4f}")

    # Join all predictions back together
    all_predictions = pd.concat(predictions).sort_index()
    
    return mean_ic, ic_scores, all_predictions


if __name__ == "__main__":
    # This assumes data_features.py has been run and dummy_data is available
    from data_features import dummy_data
    
    mean_ic, ic_scores, predictions = train_ridge_baseline(dummy_data)