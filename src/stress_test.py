# src/stress_test.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ml_utils import stress_test_model
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

def run_stress_test(dummy_data):
    logger.info("--- STARTING STRESS TEST ---")
    
    # 1. Prepare Data
    clean_data = dummy_data.dropna()
    X = clean_data.drop(columns=['target_1m', 'target_2m', 'target_3m', 'target_6m', 'target_12m'])
    y = clean_data['target_1m']
    
    # Scale Data (Important because noise is added based on scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train Model (We train once on clean data)
    logger.info("Training Reference Model...")
    model = RandomForestRegressor(
        n_estimators=500,        # Changed from 100 to match main.py
        max_depth=3,             # Same
        min_samples_leaf=20,     # Added to match main.py
        random_state=42,         # Same
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # 3. Run Stress Test
    # We test noise levels from 0.1 (small jitter) to 5.0 (market chaos)
    results = stress_test_model(model, X_scaled, y, noise_levels=[0.1, 0.5, 1.0, 2.0, 5.0])
    
    # 4. Visualize
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results, x='noise_level', y='ic', marker='o')
    plt.axhline(0, color='red', linestyle='--', label='Zero Predictivity')
    plt.title('Model Robustness: Performance vs. Market Noise')
    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel('Information Coefficient (IC)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('stress_test_results.png')
    logger.info("\nSaved chart to stress_test_results.png")
    
    return results

if __name__ == "__main__":
    from data_features import dummy_data
    run_stress_test(dummy_data)









