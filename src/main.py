"""
Main Pipeline - Orchestrates the entire quantitative strategy workflow

This script runs all components in sequence:
1. Feature engineering (data_features.py)
2. Ridge baseline model (ridge_baseline.py)
3. Feature importance analysis (feature_importance.py)
4. Portfolio signal generation (portfolio_optimizer.py)
5. Random Forest backtest (random_forest_backtest.py)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTITATIVE TRADING STRATEGY PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and engineer features
    print("\n[1/5] Loading data and engineering features...")
    from data_features import dummy_data
    print(f"✓ Data loaded: {dummy_data.shape[0]} samples, {dummy_data.shape[1]} features")
    
    # Step 2: Train Ridge baseline
    print("\n[2/5] Training Ridge Regression baseline...")
    from ridge_baseline import train_ridge_baseline
    mean_ic, ic_scores, ridge_predictions = train_ridge_baseline(dummy_data)
    print(f"✓ Ridge model trained (Mean IC: {mean_ic:.4f})")
    
    # Step 3: Analyze feature importance
    print("\n[3/5] Analyzing feature importance...")
    from feature_importance import analyze_feature_importance
    importance = analyze_feature_importance(dummy_data)
    print("✓ Feature importance calculated and saved")
    
    # Step 4: Generate current portfolio signals
    print("\n[4/5] Generating portfolio signals...")
    from portfolio_optimizer import generate_portfolio_signals
    signals = generate_portfolio_signals(dummy_data)
    print("✓ Portfolio signals generated")
    
    # Step 5: Backtest Random Forest
    print("\n[5/5] Running Random Forest backtest...")
    from random_forest_backtest import backtest_random_forest
    clean_data = dummy_data.dropna()
    returns, cumulative, sharpe, rf_predictions = backtest_random_forest(clean_data)
    print("✓ Backtest complete")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nRidge IC: {mean_ic:.4f}")
    print(f"Random Forest Sharpe: {sharpe:.2f}")
    print(f"Total Return: {(cumulative.iloc[-1] - 1):.2%}")
    print("\nGenerated files:")
    print("  - feature_importance.png")
    print("  - rf_equity_curve.png")