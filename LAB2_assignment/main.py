import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set visual style
sns.set_theme(style="whitegrid")

# --- 1. Data Loading (Handling Encoding Errors) ---
try:
    # Try default encoding for typical CSVs
    colleges_df = pd.read_csv('all_colleges_aishe.csv', encoding='latin1')
    schools_df = pd.read_csv('school_enrollment.csv', encoding='latin1')
except:
    # Fallback for Windows-generated CSVs
    colleges_df = pd.read_csv('all_colleges_aishe.csv', encoding='cp1252')
    schools_df = pd.read_csv('school_enrollment.csv', encoding='cp1252')

def clean_district(text):
    # Removes codes like " (2812)" and standardizes case
    return re.sub(r'\s*\(\d+\)', '', str(text)).strip().upper()

# --- 2. Data Preparation & Merging ---
# Clean District Names for Joining
dist_col = 'district_name & code' if 'district_name & code' in schools_df.columns else schools_df.columns[2]
schools_df['District_Clean'] = schools_df[dist_col].apply(clean_district)
colleges_df['District_Clean'] = colleges_df['District'].str.upper().str.strip()

# Aggregate Data
school_agg = schools_df.groupby('District_Clean').agg({
    'Total': 'sum',
    'School_Name': 'count'
}).rename(columns={'School_Name': 'Num_Schools'}).reset_index()

college_agg = colleges_df.groupby('District_Clean').size().reset_index(name='Num_Colleges')

# Merge to create Composite Dataset
df = pd.merge(school_agg, college_agg, on='District_Clean', how='inner')

# --- 3. Feature Engineering (Artificial Regressors) ---
np.random.seed(42)
# Create a 'Simulated_GDP' strongly correlated with Colleges (Target)
df['Simulated_GDP'] = (df['Num_Colleges'] * 50) + np.random.normal(0, 100, len(df))
# Create 'Simulated_Pop' correlated with Enrollment (Multicollinearity source)
df['Simulated_Pop'] = (df['Total'] * 2.5) + np.random.normal(0, 500, len(df))

# --- 4. Visualizations ---
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('heatmap.png')

# Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df, x='Total', y='Num_Colleges', ax=axes[0], color='skyblue')
axes[0].set_title('Enrollment vs Colleges')
sns.scatterplot(data=df, x='Simulated_GDP', y='Num_Colleges', ax=axes[1], color='coral')
axes[1].set_title('GDP vs Colleges (Strong Linear)')
sns.scatterplot(data=df, x='Simulated_Pop', y='Num_Colleges', ax=axes[2], color='lightgreen')
axes[2].set_title('Population vs Colleges')
plt.tight_layout()
plt.savefig('scatter_features.png')

# --- 5. Gradient Descent & Regularization Experiment ---
X = df[['Total', 'Num_Schools', 'Simulated_GDP', 'Simulated_Pop']]
y = df['Num_Colleges']
features = X.columns

# Standardize Data (Critical for GD)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []
models = {}
configs = [
    {'lr': 0.01, 'penalty': None, 'name': 'Standard GD'},
    {'lr': 0.1, 'penalty': 'l2', 'name': 'Ridge (L2)'},
    {'lr': 0.1, 'penalty': 'l1', 'name': 'Lasso (L1)'}
]

for config in configs:
    # SGDRegressor IS Gradient Descent in Scikit-Learn
    model = SGDRegressor(
        learning_rate='constant', eta0=config['lr'],
        penalty=config['penalty'], alpha=0.1,
        max_iter=5000, tol=1e-3, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Model': config['name'], 'MSE': mse, 'R2': r2, 'Coef': model.coef_})
    models[config['name']] = model

# Print Text Results
results_df = pd.DataFrame(results)[['Model', 'MSE', 'R2']]
print("\nExperimental Results:")
print(results_df)

# --- 6. Plotting Coefficients & Predictions ---
# Coefficient Bar Plot
coef_data = []
for res in results:
    for feat, coef in zip(features, res['Coef']):
        coef_data.append({'Model': res['Model'], 'Feature': feat, 'Coefficient': coef})

plt.figure(figsize=(12, 6))
sns.barplot(data=pd.DataFrame(coef_data), x='Feature', y='Coefficient', hue='Model', palette='viridis')
plt.title('Impact of Regularization: Lasso (L1) zeros out irrelevant features')
plt.axhline(0, color='black')
plt.savefig('coefficients.png')

# Actual vs Predicted Plot (Best Model)
plt.figure(figsize=(8, 8))
y_pred_best = models['Lasso (L1)'].predict(X_test_scaled)
sns.scatterplot(x=y_test, y=y_pred_best, s=150, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.title('Actual vs Predicted (Lasso Model)')
plt.legend()
plt.savefig('actual_vs_predicted.png')