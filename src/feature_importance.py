import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

def analyze_feature_importance(dummy_data):
    """
    Analyze feature importance using Ridge regression coefficients.
    
    Parameters:
    -----------
    dummy_data : pd.DataFrame
        Engineered feature data with targets
        
    Returns:
    --------
    pd.DataFrame : Feature importance statistics
    """
    # --- 1. CLEAN THE DATA ---
    # Drop any row that has a NaN in the features (e.g., the first 24 months of rolling windows)
    # We strictly align X and y after dropping NaNs
    clean_data = dummy_data.dropna()

    logger.info(f"Original shape: {dummy_data.shape}")
    logger.info(f"Clean shape: {clean_data.shape}")

    X = clean_data.drop(columns=['target_1m', 'target_2m', 'target_3m', 'target_6m', 'target_12m'])
    y = clean_data['target_1m']

    # Initialize Validator
    cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

    feature_names = X.columns
    coefs = []

    logger.info("Training to extract feature importance...")

    # --- 2. TRAINING LOOP ---
    for train_idx, test_idx in cv.split(clean_data):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Store coefficients
        coefs.append(model.coef_)

    # --- 3. AGGREGATE RESULTS ---
    coef_df = pd.DataFrame(coefs, columns=feature_names)

    # Calculate stats
    mean_coefs = coef_df.mean()
    std_coefs = coef_df.std()

    importance = pd.DataFrame({
        'mean_coef': mean_coefs,
        'std_coef': std_coefs
    })

    # Signal Strength = Mean / Std (How consistent is this factor?)
    importance['signal_strength'] = importance['mean_coef'].abs() / importance['std_coef']

    # Sort by impact
    importance = importance.sort_values(by='mean_coef', ascending=False)

    # --- 4. VISUALIZE ---
    plt.figure(figsize=(10, 8))
    top_features = pd.concat([importance.head(10), importance.tail(10)])

    sns.barplot(x=top_features['mean_coef'], y=top_features.index)
    plt.title('Top 20 Predictive Features (Ridge Regression)')
    plt.xlabel('Coefficient Value (Impact on 1-Month Return)')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    logger.info("\nTop 5 Positive Predictors (Buy Signals):")
    logger.info(importance.head(5)[['mean_coef', 'signal_strength']])

    logger.info("\nTop 5 Negative Predictors (Sell Signals):")
    logger.info(importance.tail(5)[['mean_coef', 'signal_strength']])
    
    return importance


if __name__ == "__main__":
    # This assumes data_features.py has been run and dummy_data is available
    from data_features import dummy_data
    
    importance = analyze_feature_importance(dummy_data)