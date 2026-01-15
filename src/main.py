import argparse
import logging
import sys
import pandas as pd
import os

# --- 1. SETUP LOGGING ---
# This looks much more professional than simple 'print' statements
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Smart data loader: Checks for cache first to save time."""
    if os.path.exists("data_cache.pkl"):
        logger.info("Loading features from cache (data_cache.pkl)...")
        return pd.read_pickle("data_cache.pkl")
    else:
        logger.info("Cache not found. Running full data pipeline...")
        from data_features import dummy_data  # This triggers the download/calc
        dummy_data.to_pickle("data_cache.pkl")
        return dummy_data

def main():
    # --- 2. COMMAND LINE INTERFACE ---
    parser = argparse.ArgumentParser(description="Quantitative Trading Strategy Pipeline")
    parser.add_argument(
        '--step', 
        type=str, 
        default='all',
        choices=['all', 'features', 'ridge', 'importance', 'portfolio', 'backtest', 'stress'],
        help='Which step of the pipeline to run'
    )
    args = parser.parse_args()

    logger.info(f"--- STARTING PIPELINE (Step: {args.step}) ---")

    # Step 1: Feature Engineering (Always needed unless just loading cache)
    if args.step == 'features':
        from data_features import dummy_data
        dummy_data.to_pickle("data_cache.pkl")
        logger.info("Features engineered and cached.")
        return # Exit if we only wanted features

    # Load data for all other steps
    data = load_data()

    # Step 2: Ridge Baseline
    if args.step in ['all', 'ridge']:
        logger.info(">>> Training Ridge Regression Baseline...")
        from ridge_baseline import train_ridge_baseline
        mean_ic, ic_scores, _ = train_ridge_baseline(data)
        logger.info(f"Ridge Mean IC: {mean_ic:.4f}")

    # Step 3: Feature Importance
    if args.step in ['all', 'importance']:
        logger.info(">>> Analyzing Feature Importance...")
        from feature_importance import analyze_feature_importance
        analyze_feature_importance(data)

    # Step 4: Portfolio Signals (Live Production)
    if args.step in ['all', 'portfolio']:
        logger.info(">>> Generating Live Portfolio Signals...")
        from portfolio_optimizer import generate_portfolio_signals
        generate_portfolio_signals(data)

    # Step 5: Random Forest Backtest
    if args.step in ['all', 'backtest']:
        logger.info(">>> Running Random Forest Backtest...")
        from random_forest_backtest import backtest_random_forest
        # Strictly remove rows with no target for backtesting
        clean_data = data.dropna(subset=['target_1m'])
        backtest_random_forest(clean_data)

    
    logger.info("--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()