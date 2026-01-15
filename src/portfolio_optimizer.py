import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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

def generate_portfolio_signals(dummy_data):
    logger.info("--- PORTFOLIO OPTIMIZER ENGINE (LIVE PRODUCTION) ---")

    # 1. SEPARATE TRAIN vs PREDICT
    # Train: Rows where we have a target (History)
    train_data = dummy_data.dropna(subset=['target_1m'])
    
    # Predict: Rows where target is NaN (The "Live" edge of data)
    # We want the very last available date
    last_date = dummy_data.index.get_level_values('date').max()
    live_data = dummy_data.xs(last_date, level='date')

    logger.info(f"Training on {len(train_data)} historical months...")
    logger.info(f"Generating signals for LIVE DATE: {last_date.date()}")

    # 2. PREPARE FEATURES
    drop_cols = ['target_1m', 'target_2m', 'target_3m', 'target_6m', 'target_12m']
    X_train = train_data.drop(columns=drop_cols, errors='ignore')
    y_train = train_data['target_1m']
    
    # Ensure Live data has same columns
    X_live = live_data.drop(columns=drop_cols, errors='ignore')
    X_live = X_live[X_train.columns] # Align columns strictly

    # 3. TRAIN MODEL
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_live_scaled = scaler.transform(X_live)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 4. PREDICT
    predictions = model.predict(X_live_scaled)

    signals = pd.DataFrame({
        'predicted_return': predictions
    }, index=X_live.index)

    signals = signals.sort_values(by='predicted_return', ascending=False)
    
    # 5. RANKING (Use Rank, not just Raw Number)
    # We add a 'Rank' column (1 = Best, 30 = Worst)
    signals['rank'] = range(1, len(signals) + 1)

    logger.info("\n" + "="*50)
    logger.info("  LIVE TRADING SIGNALS")
    logger.info("="*50)
    logger.info("\n--- TOP 5 BUYS (LONG) ---")
    logger.info(signals.head(5))

    logger.info("\n--- TOP 5 SELLS (SHORT) ---")
    logger.info(signals.tail(5))
    
    return signals

if __name__ == "__main__":
    from data_features import dummy_data
    signals = generate_portfolio_signals(dummy_data)