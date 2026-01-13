import pandas as pd
import numpy as np

# 1. LOAD AND DESCRIBE DATASET
df = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nStatistical Summary:\n", df.describe())

# 2. CHECK NULL VALUES
print("\n=== NULL VALUE ANALYSIS ===")
null_counts = df.isnull().sum()
print("Null counts per column:\n", null_counts[null_counts > 0])
print(f"Total rows with nulls: {df.isnull().any(axis=1).sum()}")

# 3. CORRELATION ANALYSIS
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Find high correlations
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Var1': correlation_matrix.columns[i],
                'Var2': correlation_matrix.columns[j],
                'Corr': correlation_matrix.iloc[i, j]
            })
            
print("\n=== HIGH CORRELATIONS (>0.7) ===")
print(pd.DataFrame(high_corr))

correlation_matrix.to_csv('correlation_matrix.csv')

# 4. OUTLIER DETECTION (IQR Method)
print("\n=== OUTLIER DETECTION ===")
outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    
    if len(outliers) > 0:
        outlier_summary.append({
            'Column': col,
            'Count': len(outliers),
            'Percentage': f"{(len(outliers)/len(df)*100):.2f}%",
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })

print(pd.DataFrame(outlier_summary))

# 5. CHANGE DATA TYPES
df_modified = df.copy()
df_modified['Reporting Year'] = df_modified['Reporting Year'].astype(str)
df_modified['Reporting Month'] = df_modified['Reporting Month'].astype(str)
df_modified['AWC_ID'] = df_modified['AWC_ID'].astype(str)

print("\n=== DATA TYPE CHANGES ===")
print("Original:", df[['Reporting Year', 'AWC_ID']].dtypes)
print("Modified:", df_modified[['Reporting Year', 'AWC_ID']].dtypes)

# 6. REGRESSION PROBLEM SETUP
df_regression = df.copy()

# Create target variable
df_regression['Total_Children_Enrolled'] = (
    df_regression['Tot_Children_0to6M'] + 
    df_regression['Tot_Children_7Mto3Y'] + 
    df_regression['Tot_Children_3Yto6Y']
)

# Create aggregate features
df_regression['Total_SC'] = (
    df_regression['Tot_Children_SC_0to6M'] + 
    df_regression['Tot_Children_SC_7Mto3Y'] + 
    df_regression['Tot_Children_SC_3Yto6Y']
)

df_regression['Total_ST'] = (
    df_regression['Tot_Children_ST_0to6M'] + 
    df_regression['Tot_Children_ST_7Mto3Y'] + 
    df_regression['Tot_Children_ST_3Yto6Y']
)

df_regression['Total_BC'] = (
    df_regression['Tot_Children_BC_0to6M'] + 
    df_regression['Tot_Children_BC_7Mto3Y'] + 
    df_regression['Tot_Children_BC_3Yto6Y']
)

df_regression['Total_OC'] = (
    df_regression['Tot_Children_OC_0to6M'] + 
    df_regression['Tot_Children_OC_7Mto3Y'] + 
    df_regression['Tot_Children_OC_3Yto6Y']
)

print("\n=== REGRESSION SETUP ===")
print("Target: Total_Children_Enrolled")
print(f"Mean: {df_regression['Total_Children_Enrolled'].mean():.2f}")
print(f"Features: Total_SC, Total_ST, Total_BC, Total_OC, D_Name, Proj_Name")

df_regression.to_csv('regression_dataset.csv', index=False)

# 7. JOIN WITH EXTERNAL DATASET (Example)
# Create simulated external data
np.random.seed(42)
unique_ids = df['AWC_ID'].unique()[:1000]

external_df = pd.DataFrame({
    'AWC_ID': unique_ids,
    'Staff_Count': np.random.randint(2, 8, len(unique_ids)),
    'Budget': np.random.randint(50000, 200000, len(unique_ids))
})

# Perform left join
df_joined = df.merge(external_df, on='AWC_ID', how='left')

print("\n=== DATASET JOIN ===")
print(f"Original shape: {df.shape}")
print(f"External shape: {external_df.shape}")
print(f"Joined shape: {df_joined.shape}")
print(f"Nulls after join: {df_joined['Staff_Count'].isnull().sum()}")

df_joined.to_csv('joined_dataset.csv', index=False)

print("\n=== FILES CREATED ===")
print("- correlation_matrix.csv")
print("- regression_dataset.csv")
print("- joined_dataset.csv")
