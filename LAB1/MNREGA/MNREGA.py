import pandas as pd
import numpy as np

# 1. LOAD AND DESCRIBE DATASET
df = pd.read_csv('MGNREGA_dataset_AtAGlance.csv')

print("="*80)
print("DATASET DESCRIPTION")
print("="*80)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistical Summary:\n{df.describe()}")

# 2. NULL VALUE ANALYSIS
print("\n" + "="*80)
print("NULL VALUE ANALYSIS")
print("="*80)
null_counts = df.isnull().sum()
print(f"Columns with nulls:\n{null_counts[null_counts > 0]}")
print(f"Total rows with nulls: {df.isnull().any(axis=1).sum()}")

# 3. CORRELATION ANALYSIS
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

print("\n" + "="*80)
print("HIGH CORRELATIONS (>0.7)")
print("="*80)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Var1': corr_matrix.columns[i],
                'Var2': corr_matrix.columns[j],
                'Corr': corr_matrix.iloc[i, j]
            })
            
print(pd.DataFrame(high_corr).sort_values('Corr', ascending=False))
corr_matrix.to_csv('mgnrega_correlation_matrix.csv')

# 4. OUTLIER DETECTION (IQR METHOD)
print("\n" + "="*80)
print("OUTLIER DETECTION")
print("="*80)

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
            'Lower': lower,
            'Upper': upper,
            'Min': df[col].min(),
            'Max': df[col].max()
        })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)
outlier_df.to_csv('mgnrega_outlier_analysis.csv', index=False)

# 5. CHANGE DATA TYPES
df_modified = df.copy()
df_modified['state_code'] = df_modified['state_code'].astype(str)
df_modified['district_code'] = df_modified['district_code'].astype(str)

print("\n" + "="*80)
print("DATA TYPE CHANGES")
print("="*80)
print("Original:", df[['state_code', 'district_code']].dtypes)
print("Modified:", df_modified[['state_code', 'district_code']].dtypes)

# 6. REGRESSION PROBLEM SETUP
df_reg = df.copy()

# Create engineered features
df_reg['active_worker_ratio'] = (
    df_reg['total_no_of_active_workers'] / df_reg['total_no_of_workers']
)

df_reg['job_card_activation_rate'] = (
    df_reg['total_no_of_active_job_cards'] / 
    df_reg['total_no_of_job_cards_issued']
)

df_reg['work_completion_rate'] = (
    df_reg['no_of_completed_works'] / 
    (df_reg['total_no_of_works_taken_up_new_and_spillover'] + 1)
)

df_reg['wage_per_personday'] = (
    df_reg['wages'] / (df_reg['persondays_of_central_liability_so_far'] + 1)
)

print("\n" + "="*80)
print("REGRESSION PROBLEM")
print("="*80)
print("Target: average_days_employment_per_household")
print(f"Mean: {df_reg['average_days_employment_per_household'].mean():.2f}")
print(f"Std: {df_reg['average_days_employment_per_household'].std():.2f}")
print("\nFeatures: active_worker_ratio, job_card_activation_rate,")
print("          work_completion_rate, total_expenditure, wages")

# Save regression dataset
reg_cols = ['state_name', 'district_name', 'total_no_of_active_workers',
            'total_households_worked', 'approved_labour_budget',
            'total_expenditure', 'wages', 'active_worker_ratio',
            'job_card_activation_rate', 'work_completion_rate',
            'average_days_employment_per_household']
df_reg[reg_cols].to_csv('mgnrega_regression_dataset.csv', index=False)

# 7. JOIN WITH EXTERNAL DATASET (Anganwadi)
df_anganwadi = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')

# Aggregate at district level
ang_district = df_anganwadi.groupby('D_Name').agg({
    'Tot_Children_0to6M': 'sum',
    'Tot_Children_7Mto3Y': 'sum',
    'Tot_Children_3Yto6Y': 'sum',
    'AWC_ID': 'count'
}).reset_index()

ang_district['Total_Children'] = (
    ang_district['Tot_Children_0to6M'] + 
    ang_district['Tot_Children_7Mto3Y'] + 
    ang_district['Tot_Children_3Yto6Y']
)

ang_district.rename(columns={'D_Name': 'district_name', 
                             'AWC_ID': 'Total_AWCs'}, inplace=True)

# Normalize district names
df['district_normalized'] = df['district_name'].str.upper().str.strip()
ang_district['district_normalized'] = ang_district['district_name'].str.upper().str.strip()

# Join
df_joined = df.merge(
    ang_district[['district_normalized', 'Total_AWCs', 'Total_Children']], 
    on='district_normalized', 
    how='left'
)

print("\n" + "="*80)
print("DATASET JOIN")
print("="*80)
print(f"MGNREGA shape: {df.shape}")
print(f"Anganwadi districts: {ang_district.shape}")
print(f"Joined shape: {df_joined.shape}")
print(f"Unmatched rows: {df_joined['Total_AWCs'].isnull().sum()}")

df_joined.to_csv('mgnrega_anganwadi_joined.csv', index=False)

print("\n" + "="*80)
print("FILES CREATED")
print("="*80)
print("1. mgnrega_correlation_matrix.csv")
print("2. mgnrega_outlier_analysis.csv")
print("3. mgnrega_regression_dataset.csv")
print("4. mgnrega_anganwadi_joined.csv")
