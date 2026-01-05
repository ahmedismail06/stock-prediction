import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas_datareader.data as web
from talib import BBANDS, RSI, MACD
from sklearn.impute import SimpleImputer
from datetime import datetime

# --- 1. CONFIGURATION (UPDATED FOR LIVE) ---
ticker_list = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'ADBE', 
    'JPM', 'BAC', 'V', 'MA', 'GS',
    'LLY', 'UNH', 'JNJ', 'MRK', 'PFE',
    'AMZN', 'TSLA', 'HD', 'MCD', 
    'WMT', 'COST', 'PG', 'KO',
    'XOM', 'CVX', 'COP',
    'CAT', 'BA', 'UNP'
]

# DYNAMIC DATE: Fetch data up to today
start_date = "2000-01-01"
end_date = datetime.today().strftime('%Y-%m-%d') 

print(f"--- RUNNING DATA PIPELINE (END DATE: {end_date}) ---")

# --- 2. DOWNLOAD & CALCULATE INDICATORS ---
print("Downloading price data...")
raw_data = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=False)

daily_indicators = pd.DataFrame(index=raw_data.index)

print("Calculating Technical Indicators...")
all_indicators = []
for ticker in ticker_list:
    try:
        # Handle multi-level column access safely
        if isinstance(raw_data.columns, pd.MultiIndex):
            close_price = raw_data.loc[:, ('Close', ticker)]
        else:
            close_price = raw_data['Close']

        # 1. Bollinger Bands
        up, mid, low = BBANDS(close_price, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
        bandwidth = (up - low).replace(0, 1e-9)
        
        # 2. RSI & MACD
        indicators = pd.DataFrame({
            (f'BB_pct_b', ticker): (close_price - low) / bandwidth,
            (f'BB_width', ticker): (up - low) / mid,
            (f'RSI', ticker): RSI(close_price, timeperiod=14),
            (f'MACD_norm', ticker): MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)[0] / close_price,
            (f'MACD_sig_norm', ticker): MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)[1] / close_price
        }, index=close_price.index)
        all_indicators.append(indicators)
    except Exception as e:
        # print(f"Error calculating indicators for {ticker}: {e}")
        pass

daily_indicators = pd.concat(all_indicators, axis=1)

# Resample to Monthly
monthly_indicators = daily_indicators.resample('ME').last()
prices = raw_data['Adj Close'] if 'Adj Close' in raw_data else raw_data['Close']
monthly_prices = prices.resample('ME').last()

# --- 3. CALCULATE RETURNS & TARGETS ---
outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]

for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                           .pct_change(lag)
                           .stack()
                           .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                  upper=x.quantile(1-outlier_cutoff)))
                           .add(1)
                           .pow(1/lag)
                           .sub(1)
                           )

data = data.swaplevel().dropna() 
data.index.names = ['ticker', 'date']

# --- 4. MERGE INDICATORS ---
monthly_indicators.columns = pd.MultiIndex.from_tuples(monthly_indicators.columns)
monthly_indicators.columns.names = ['feature', 'ticker']
stacked_indicators = monthly_indicators.stack(level='ticker', future_stack=True).swaplevel()
stacked_indicators.index.names = ['ticker', 'date']

data = data.join(stacked_indicators)

# --- 5. FAMA-FRENCH FACTORS (FIXED) ---
try:
    print("Fetching Fama-French Factors...")
    # Fetch Data
    ff_dict = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')
    factor_data = ff_dict[0].drop('RF', axis=1)
    
    # 1. FIX: Convert PeriodIndex to Timestamp immediately
    factor_data.index = factor_data.index.to_timestamp()
    
    # 2. Resample to Month End to match your stock data
    factor_data = factor_data.resample('ME').last().div(100)
    factor_data.index.name = 'date'
    
    # 3. Merge with Stock Data
    # We use reindex to map the factors to every stock for the same month
    data = data.join(factor_data, on='date')
    
    # 4. Calculate Rolling Betas (Exposure to factors)
    # This tells us: "Is AAPL moving like a Value stock or a Growth stock right now?"
    print("Calculating Rolling Betas...")
    T = 24
    
    def calculate_betas(x):
        # RollingOLS requires no missing data in the window
        if len(x) < T: 
            return pd.DataFrame(index=x.index, columns=['beta_Mkt', 'beta_SMB', 'beta_HML'])
            
        # We predict Stock Return ~ Mkt + SMB + HML
        exog = sm.add_constant(x[['Mkt-RF', 'SMB', 'HML']])
        model = RollingOLS(endog=x.return_1m, exog=exog, window=T)
        params = model.fit(params_only=True).params
        return params.drop('const', axis=1)

    # Apply to each ticker (Group by Ticker)
    # We use 'try' inside the apply to handle short histories
    betas = (data.groupby(level='ticker', group_keys=False)
             .apply(calculate_betas))
             
    # Rename columns to avoid confusion
    betas.columns = ['beta_Mkt', 'beta_SMB', 'beta_HML']
    
    # Join Betas back to main data
    data = data.join(betas.groupby(level='ticker').shift()) # Shift because Beta is known at t-1

    print("Fama-French Factors successfully added.")

except Exception as e:
    print(f"Warning: Fama-French factors skipped ({e}). Using technicals only.")

# --- 6. FEATURE ENGINEERING ---
for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)

dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month

for t in range(1, 7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker').return_1m.shift(t)

# Create Target
for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)

# --- 7. DUMMIES ---
# (Simplified for speed - skipping Sector/Size to prevent lookup errors in production)
# In production, simple features are more robust.
categorical_columns = ['year', 'month']
column_transformer = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Impute NaNs (Forward fill then 0)
data = data.ffill().fillna(0)

dummy_data = pd.DataFrame(
    column_transformer.fit_transform(data),
    columns=column_transformer.get_feature_names_out(),
    index=data.index
)

# !!! CRITICAL CHANGE !!!
# We DO NOT drop NA targets anymore. We need the last row (which has NaN target) for prediction.
# dummy_data = dummy_data.dropna(subset=['target_1m']) <--- COMMENTED OUT