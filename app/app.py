import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

import os

# CONFIGURATION

DATA_PATH = r'C:\Users\kelec\source\repos\CUSTOMER-SEGMENTATION\Online Retail.xlsx'  
OUTPUT_PATH = './data/processed/'
RAW_DATA_PATH = './data/raw/'

# Create directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# LOAD DATA
print("Loading data...")
df = pd.read_excel(DATA_PATH, sheet_name='Online Retail')
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# DATA CLEANING
print("\n--- DATA CLEANING ---")

# Remove rows with missing CustomerID (critical for segmentation)
print(f"Rows with missing CustomerID: {df['CustomerID'].isna().sum()}")
df = df.dropna(subset=['CustomerID'])

# Remove rows with missing InvoiceDate
df = df.dropna(subset=['InvoiceDate'])

# Remove rows with negative Quantity (returns/cancellations)
print(f"Rows with negative Quantity: {(df['Quantity'] < 0).sum()}")
df = df[df['Quantity'] > 0]

# Remove rows with zero or negative UnitPrice
print(f"Rows with non-positive UnitPrice: {(df['UnitPrice'] <= 0).sum()}")
df = df[df['UnitPrice'] > 0]

print(f"✓ After cleaning: {len(df)} rows")

# CALCULATE TOTAL SPENT PER TRANSACTION
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']
print(f"✓ Created TotalSpent column")

# CONVERT DATE TO DATETIME
# Excel stores dates as numbers, convert them
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], unit='D', origin='1899-12-30')
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

# Reference date for RFM calculation (last date + 1 day)
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Reference date for RFM: {reference_date}")

# CREATE RFM FEATURES (Recency, Frequency, Monetary)
# Group by CustomerID
customer_data = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',           # Frequency (number of transactions)
    'TotalSpent': 'sum',              # Monetary (total amount spent)
    'InvoiceDate': 'max',             # Last purchase date
    'Quantity': 'sum',                # Total items purchased
    'Country': 'first'                # Customer country
}).reset_index()

# Rename columns for clarity
customer_data.columns = ['CustomerID', 'Frequency', 'Monetary', 'LastPurchaseDate', 'TotalQuantity', 'Country']

# Calculate Recency (days since last purchase)
customer_data['Recency'] = (reference_date - customer_data['LastPurchaseDate']).dt.days

# Calculate additional features
customer_data['AvgOrderValue'] = customer_data['Monetary'] / customer_data['Frequency']
customer_data['ItemsPerTransaction'] = customer_data['TotalQuantity'] / customer_data['Frequency']

print(f"✓ Created {len(customer_data)} customer profiles")
print(f"\nRFM Statistics:")
print(customer_data[['Recency', 'Frequency', 'Monetary']].describe())

# SELECT FEATURES FOR CLUSTERING
print("\n--- SELECTING FEATURES FOR CLUSTERING ---")

# Features to use for clustering
clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'ItemsPerTransaction']

X = customer_data[clustering_features].copy()

print(f"Features: {clustering_features}")
print(f"Feature shape: {X.shape}")

# HANDLE OUTLIERS (Optional - using IQR method)
print("\n--- HANDLING OUTLIERS ---")

outliers_before = len(X)

for col in clustering_features:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
    
    print(f"{col}: removed {outliers_before - len(X)} outliers (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    outliers_before = len(X)

# Remove corresponding customer rows
customer_data = customer_data.loc[X.index]

print(f"✓ After outlier removal: {len(X)} customers")

# NORMALIZE FEATURES
print("\n--- NORMALIZING FEATURES ---")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=clustering_features, index=X.index)

print(f"✓ Features normalized (StandardScaler)")
print(f"Scaled data statistics:")
print(X_scaled.describe())

# SAVE PROCESSED DATA
# Save customer data with features
customer_data.to_csv(f'{OUTPUT_PATH}customer_profiles.csv', index=False)

# Save scaled features
X_scaled.to_csv(f'{OUTPUT_PATH}scaled_features.csv')

# Save original features for reference
X.to_csv(f'{OUTPUT_PATH}original_features.csv')

# Save scaler for later use
joblib.dump(scaler, f'{OUTPUT_PATH}scaler.pkl')

# Save raw data for reference
df.to_csv(f'{OUTPUT_PATH}raw_transactions.csv', index=False)

# SUMMARY STATISTICS
print("\n" + "="*60)
print("DATA PREPARATION SUMMARY")
print("="*60)
print(f"Total transactions: {len(df)}")
print(f"Total customers: {len(customer_data)}")
print(f"Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
print(f"Countries: {customer_data['Country'].nunique()}")
print(f"\nTop 5 countries:")
print(customer_data['Country'].value_counts().head())
print(f"\nMonetary Statistics:")
print(f"  Total Revenue: £{customer_data['Monetary'].sum():,.2f}")
print(f"  Average Customer Value: £{customer_data['Monetary'].mean():,.2f}")
print(f"  Median Customer Value: £{customer_data['Monetary'].median():,.2f}")
print(f"  Max Customer Value: £{customer_data['Monetary'].max():,.2f}")
print(f"\nFrequency Statistics:")
print(f"  Average Transactions per Customer: {customer_data['Frequency'].mean():.1f}")
print(f"  Max Transactions: {customer_data['Frequency'].max()}")