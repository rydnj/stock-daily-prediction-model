import numpy as np
import pandas as pd

# Load preprocessed data
data = pd.read_csv('preprocessed_AAPL.csv')

# Create new features
data['Moving_Average_10'] = data['close'].rolling(window=10).mean()
data['Moving_Average_50'] = data['close'].rolling(window=50).mean()
data['Volatility'] = data['close'].rolling(window=10).std()
data['Momentum'] = data['close'] - data['close'].shift(10)

# Handle missing values created by rolling calculations
data = data.dropna()

# Save data with new features
data.to_csv('feature_engineered_AAPL.csv', index=False)
