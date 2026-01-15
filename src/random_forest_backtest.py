import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

def backtest_random_forest(clean_data, long_top_n=5, short_top_n=5):
    """
    Backtest Random Forest strategy with long/short portfolio.
    
    Parameters:
    -----------
    clean_data : pd.DataFrame
        Cleaned engineered feature data
    long_top_n : int
        Number of stocks to long
    short_top_n : int
        Number of stocks to short
        
    Returns:
    --------
    tuple : (portfolio_returns, cumulative_return, sharpe_ratio, all_predictions)
    """
    # --- 1. SETUP ---
    X = clean_data.drop(columns=['target_1m', 'target_2m', 'target_3m', 'target_6m', 'target_12m'])
    y = clean_data['target_1m']

    # --- 2. RANDOM FOREST BACKTEST ---
    cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
    predictions = []

    logger.info("Running Random Forest Simulation (This may take a minute)...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(clean_data), 1):
        logger.info(f"  Processing Fold {fold}...")
        
        # Split
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Note: Trees don't strictly require scaling, but it helps convergence 
        # and doesn't hurt, so we keep it for consistency.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- THE UPGRADE ---
        # n_estimators=100: Build 100 trees
        # max_depth=3: Keep trees shallow (prevents overfitting noise)
        # min_samples_leaf=20: Don't make a rule for just 1-2 data points
        # n_jobs=-1: Use all CPU cores (faster)
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        pred = model.predict(X_test_scaled)
        
        # Store
        fold_preds = pd.DataFrame({
            'predicted_return': pred,
            'actual_return': y_test
        }, index=y_test.index)
        
        predictions.append(fold_preds)

    # Combine
    all_predictions = pd.concat(predictions).sort_index()

    # --- 3. CALCULATE PERFORMANCE ---
    def calculate_monthly_pnl(group):
        group = group.sort_values(by='predicted_return', ascending=False)
        longs = group.head(long_top_n)
        shorts = group.tail(short_top_n)
        return longs['actual_return'].mean() - shorts['actual_return'].mean()

    portfolio_returns = all_predictions.groupby(level='date').apply(calculate_monthly_pnl)

    cumulative_return = (1 + portfolio_returns).cumprod()
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12)

    logger.info("\n--- RANDOM FOREST RESULTS ---")
    logger.info(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Total Return: {(cumulative_return.iloc[-1] - 1):.2%}")
    logger.info(f"Best Month: {portfolio_returns.max():.2%}")
    logger.info(f"Worst Month: {portfolio_returns.min():.2%}")

    # Plot
    plt.figure(figsize=(12, 6))
    cumulative_return.plot(title=f'Random Forest Strategy (Sharpe: {sharpe_ratio:.2f})')
    plt.ylabel('Growth of $1')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_equity_curve.png')
    plt.show()
    logger.info("Saved chart to rf_equity_curve.png")
    
    return portfolio_returns, cumulative_return, sharpe_ratio, all_predictions


if __name__ == "__main__":
    # This assumes data_features.py has been run and dummy_data is available
    from data_features import dummy_data
    
    # Clean the data
    clean_data = dummy_data.dropna()
    
    # Run backtest
    returns, cumulative, sharpe, predictions = backtest_random_forest(clean_data)