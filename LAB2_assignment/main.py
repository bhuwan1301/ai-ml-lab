import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Data Loading & Cleaning ---

# FIX: Added encoding='latin1' to handle special characters
try:
    colleges_df = pd.read_csv('all_colleges_aishe.csv', encoding='latin1')
    schools_df = pd.read_csv('school_enrollment.csv', encoding='latin1')
except UnicodeDecodeError:
    # Fallback if latin1 fails
    colleges_df = pd.read_csv('all_colleges_aishe.csv', encoding='cp1252')
    schools_df = pd.read_csv('school_enrollment.csv', encoding='cp1252')

def clean_district(text):
    # Converts "VIZIANAGARAM (2812)" -> "VIZIANAGARAM"
    return re.sub(r'\s*\(\d+\)', '', str(text)).strip().upper()

# Prepare School Data (Aggregating by District)
# Ensure district column exists or find the closest match if name varies
if 'district_name & code' in schools_df.columns:
    schools_df['District_Clean'] = schools_df['district_name & code'].apply(clean_district)
else:
    # Fallback if column names are different in your local file
    schools_df['District_Clean'] = schools_df.iloc[:, 2].apply(clean_district)

school_agg = schools_df.groupby('District_Clean').agg({
    'Total': 'sum',           # Regressor 1: Total Enrollment
    'School_Name': 'count'    # Regressor 2: Number of Schools
}).rename(columns={'School_Name': 'Num_Schools'}).reset_index()

# Prepare College Data (Counting Colleges per District)
colleges_df['District_Clean'] = colleges_df['District'].str.upper().str.strip()
college_agg = colleges_df.groupby('District_Clean').size().reset_index(name='Num_Colleges')

# Merge to create Composite Dataset
df = pd.merge(school_agg, college_agg, on='District_Clean', how='inner')

if df.empty:
    print("Error: Merged dataset is empty! Check if District names match between files.")
else:
    # --- 2. Feature Engineering (Artificial Regressors) ---
    np.random.seed(42)
    # Create a 'Simulated_GDP' strongly correlated with Colleges (Target)
    df['Simulated_GDP'] = (df['Num_Colleges'] * 50) + np.random.normal(0, 100, len(df))
    # Create 'Simulated_Pop' correlated with Enrollment (Multicollinearity source)
    df['Simulated_Pop'] = (df['Total'] * 2.5) + np.random.normal(0, 500, len(df))

    print(f"Dataset Shape: {df.shape}")

    # --- 3. Analysis & Visualization ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix: Identification of Multicollinearity")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png') 
    print("Saved correlation_heatmap.png")

    # --- 4. Gradient Descent Experimentation ---
    # Define X (Regressors) and y (Output)
    X = df[['Total', 'Num_Schools', 'Simulated_GDP', 'Simulated_Pop']]
    y = df['Num_Colleges']

    # Split 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling is MANDATORY for Gradient Descent
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def run_gd_experiment(lr, penalty_type):
        # SGDRegressor implements Gradient Descent
        model = SGDRegressor(learning_rate='constant', eta0=lr, 
                             penalty=penalty_type, alpha=0.1, 
                             max_iter=5000, tol=1e-3, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2, model.coef_

    # Run Loop
    print(f"\n{'LR':<6} {'Reg':<6} {'MSE':<10} {'R2':<8} {'Coefficients (Weights)'}")
    print("-" * 65)

    configs = [(0.01, None), (0.1, None), (0.01, 'l2'), (0.1, 'l1')]

    for lr, reg in configs:
        mse, r2, coefs = run_gd_experiment(lr, reg)
        reg_str = str(reg) if reg else "None"
        # Formatting coefficients to show sparsity
        coef_str = ", ".join([f"{c:.2f}" for c in coefs])
        print(f"{lr:<6} {reg_str:<6} {mse:<10.2f} {r2:<8.3f} [{coef_str}]")

    print("-" * 65)
    print("Note: In L1 (last row), look for coefficients that became 0.00")