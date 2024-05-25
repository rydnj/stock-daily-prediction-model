import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('AAPL.csv')

# Handle any missing values if present, even if it isnt there a good practise nonetheless.
data = data.dropna()

# Normalize the the data using an appropriate scaler.
scaler = StandardScaler()
# Note down the features as denoted in the datafile to scale.
numerical_features = ['open', 'high', 'low', 'close', 'volume']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save the preprocessed data file.
data.to_csv('preprocessed_AAPL.csv', index=False)