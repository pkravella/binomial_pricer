import pandas as pd
import numpy as np
import math
import os
import glob

# 1. American Binomial Tree Pricing Engine (CRR Model)
def pricing_engine(S0, K, T, r, sigma, N=100, option_type='call'):
    # Safety checks for boundary conditions
    if T <= 0:
        return max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
    if sigma <= 0.00001:
        # Risk-neutral price with no volatility
        forward_price = S0 * math.exp(r * T)
        return max(forward_price - K, 0) * math.exp(-r * T) if option_type == 'call' else max(K - forward_price, 0) * math.exp(-r * T)

    dt = T / N
    df = math.exp(-r * dt)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Initialize terminal values at the final nodes (N)
    st_prices = S0 * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
    
    if option_type == 'call':
        values = np.maximum(st_prices - K, 0)
    else:
        values = np.maximum(K - st_prices, 0)

    # Backward induction through the tree
    for i in range(N - 1, -1, -1):
        values = (p * values[:-1] + (1 - p) * values[1:]) * df
        # Re-calculate underlying prices at this level for early exercise check
        st_prices = S0 * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1, 1))
        
        if option_type == 'call':
            exercise = np.maximum(st_prices - K, 0)
        else:
            exercise = np.maximum(K - st_prices, 0)
        
        values = np.maximum(values, exercise)
        
    return values[0]

# 2. Implied Volatility Solver (Bisection Method)
def solve_iv(market_price, S0, K, T, r, opt_type='call'):
    low, high = 0.0001, 4.0 # Increased high bound for volatile markets
    for i in range(40):
        mid = (low + high) / 2
        price = pricing_engine(S0, K, T, r, mid, 100, opt_type)
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# --- USER INPUTS ---
KNOWN_MARKET_OPTION_PRICE = 67.87 
KNOWN_OPTION_STRIKE = 575.0
KNOWN_OPTION_EXPIRY_DAYS = 30
KNOWN_OPTION_TYPE = 'call'

TARGET_STRIKE = 575.0
TARGET_EXPIRY_DAYS = 0
TARGET_TYPE = 'call'

RISK_FREE_RATE = 0.045
FOLDER_PATH = './databento/'  

# --- EXECUTION ---
all_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.csv")))
current_spot = None

print(f"Reading {len(all_files)} files...")

for file in all_files:
    try:
        df = pd.read_csv(file, usecols=['bid_px_00', 'ask_px_00'])
        if not df.empty:
            latest = df.iloc[-1]
            # Calculating mid-price from the provided Databento columns
            current_spot = (latest['bid_px_00'] + latest['ask_px_00']) / 2
    except Exception:
        continue

if current_spot:
    iv = solve_iv(KNOWN_MARKET_OPTION_PRICE, current_spot, KNOWN_OPTION_STRIKE, 
                  KNOWN_OPTION_EXPIRY_DAYS/365, RISK_FREE_RATE, KNOWN_OPTION_TYPE)
    
    target_price = pricing_engine(current_spot, TARGET_STRIKE, TARGET_EXPIRY_DAYS/365, 
                                  RISK_FREE_RATE, iv, 100, TARGET_TYPE)

    print(f"\nResults:\n{'-'*20}")
    print(f"Latest SPY Spot:  ${current_spot:.2f}")
    print(f"Calibrated IV:    {iv:.2%}")
    print(f"Target Option Price (${TARGET_STRIKE} {TARGET_TYPE}): ${target_price:.2f}")
else:
    print("Error: Could not determine spot price. Check if .csv files contain bid/ask columns.")