# Stock Prediction - Quantitative Trading Strategy

A quantitative trading strategy using machine learning and technical analysis to predict stock returns and generate portfolio signals.

## Features

- **Technical Indicators**: Bollinger Bands, RSI, MACD normalized and optimized for machine learning
- **Fama-French Factors**: Integration of 5-factor model for risk decomposition
- **Feature Engineering**: Momentum indicators, rolling betas, and temporal features
- **Ridge Regression**: Baseline model with Information Coefficient (IC) validation
- **Random Forest Backtest**: Out-of-sample backtesting with long/short portfolio strategy
- **Feature Importance**: Analysis of predictive signals across engineered features
- **Portfolio Optimization**: Real-time signal generation for current market conditions

## Project Structure

```
src/
├── data_features.py           # Feature engineering and data pipeline
├── ridge_baseline.py          # Ridge regression baseline model
├── feature_importance.py      # Feature importance analysis
├── random_forest_backtest.py  # Random Forest backtesting engine
├── portfolio_optimizer.py     # Portfolio signal generation
├── ml_utils.py               # Cross-validation utilities (PurgedKFold)
└── main.py                   # Pipeline orchestration
```

## Installation

```bash
pip install pandas yfinance scikit-learn statsmodels talib pandas-datareader
```

## Usage

Run the complete pipeline:

```bash
python src/main.py
```

This executes in sequence:
1. Feature engineering (30 stocks across tech, finance, healthcare, consumer, energy, industrials)
2. Ridge regression baseline training with IC evaluation
3. Feature importance analysis
4. Portfolio signal generation
5. Random Forest backtest with performance metrics

## Technical Stack

- **Data**: yfinance, pandas-datareader
- **ML**: scikit-learn (Ridge, RandomForest)
- **Stats**: statsmodels (RollingOLS, Spearman correlation)
- **Tech Analysis**: TA-Lib

## Key Models

### Ridge Regression
- Cross-validation: PurgedKFold (5 folds, 1% embargo)
- Target: 1-month forward returns
- Evaluation: Information Coefficient (Spearman rank correlation)

### Random Forest
- Trees: 500 estimators
- Depth: 3 (prevents overfitting)
- Min samples per leaf: 20
- Strategy: Long top 5 / Short bottom 5 stocks by predicted return

## Data Pipeline

1. Download OHLC prices for 30 stocks (2000-2023)
2. Calculate daily technical indicators
3. Resample to monthly frequency
4. Compute forward-looking returns (1, 2, 3, 6, 12 months)
5. Fetch Fama-French 5-factor model
6. Engineer features: momentum, rolling betas, lagged returns
7. One-hot encode categorical features (year, month)
8. Generate target labels

## Performance Considerations

- Purged K-Fold prevents look-ahead bias
- Embargo windows protect against data leakage
- Normalized indicators for multi-scale compatibility
- Defragmented DataFrame operations for efficiency

## Future Enhancements

- Ensemble models (XGBoost, LightGBM)
- Real-time data ingestion
- Transaction cost modeling
- Risk management and position sizing
- Multiple timeframe analysis

## License

MIT
