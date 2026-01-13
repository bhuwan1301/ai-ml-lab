## Project Overview

This repo demonstrates a small but complete applied data-science workflow:

- Exploring and documenting tabular datasets  
- Checking correlations between attributes  
- Conceptualizing regression problems  
- Joining related datasets  
- Handling missing values and outliers  
- Implementing everything in reproducible Python scripts  

There are two experiments:

1. Anganwadi Children Enrollment (Dataset 1)  
2. MGNREGA At-a-Glance (Dataset 2)  

***

## Repository Structure

```text
.
├── data/
│   ├── number_of_children_enrolled_in_anganwadis_2025_10.csv
│   └── MGNREGA_dataset_AtAGlance.csv
├── outputs/
│   ├── anganwadi/
│   │   ├── correlation_matrix.csv
│   │   ├── outlier_analysis.csv
│   │   ├── regression_dataset.csv
│   │   └── joined_dataset_example.csv
│   └── mgnrega/
│       ├── mgnrega_correlation_matrix.csv
│       ├── mgnrega_outlier_analysis.csv
│       ├── mgnrega_regression_dataset.csv
│       └── mgnrega_anganwadi_joined.csv
├── notebooks/  (optional if you use .ipynb)
│   ├── anganwadi_experiment.ipynb
│   └── mgnrega_experiment.ipynb
├── scripts/
│   ├── analyze_anganwadi.py
│   └── analyze_mgnrega.py
└── README.md
```

Feel free to rename `scripts/` to `src/` or merge everything into notebooks if you prefer.

***

## Experiment 1: Anganwadi Children Enrollment

### Dataset

- File: `data/number_of_children_enrolled_in_anganwadis_2025_10.csv`  
- Size: 35,781 rows × 22 columns  
- Granularity: One row ≈ one Anganwadi Centre (AWC) for a month/year combination  

Key columns:  

- `Reporting Year`, `Reporting Month` – time period  
- `D_Name`, `Proj_Name`, `Sec_NAme` – district / project / sector  
- `AWC_ID`, `AWc_Name` – AWC identifiers  
- Age-group wise totals (0–6 months, 7 months–3 years, 3–6 years), each split by caste:
  - `Tot_Children_0to6M`, `Tot_Children_SC_0to6M`, `Tot_Children_ST_0to6M`, `Tot_Children_BC_0to6M`, `Tot_Children_OC_0to6M`  
  - Similar pattern for `7Mto3Y` and `3Yto6Y`  

### Analysis Goals

1. Describe the dataset  
   - Print shape, dtypes, head, and descriptive statistics.  

2. Correlation between attributes  
   - Build correlation matrix on numeric columns.  
   - Identify strongly correlated pairs (|corr| > 0.7), especially across age groups and caste categories.  

3. Conceptualize a regression problem  
   - Target: `Total_Children_Enrolled` = sum of all age-group totals.  
   - Features: aggregated caste-wise totals, district/project, age-group specific counts.  

4. Join with another dataset  
   - Example: join with a synthetic AWC-level dataset containing staff/infrastructure/budget details on `AWC_ID`.  

5. Change attribute types  
   - Convert identifiers like `Reporting Year`, `Reporting Month`, `AWC_ID` from numeric to categorical (string) where appropriate.  

6. Missing values  
   - Detect and summarize nulls per column and per row.  
   - For this dataset, there are no null values.  

7. Outlier detection  
   - Use IQR-based rule for each numeric column.  
   - Summarize count, percentage, and bounds for outliers per column.  

### How to Run

```bash
# Create venv (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run Anganwadi analysis
python scripts/analyze_anganwadi.py
```

Expected outputs (written to `outputs/anganwadi/`):  

- `correlation_matrix.csv` – full correlation matrix  
- `outlier_analysis.csv` – IQR-based outlier summary  
- `regression_dataset.csv` – engineered regression-ready dataset  
- `joined_dataset_example.csv` – example join with an external AWC-level dataset  

***

## Experiment 2: MGNREGA At-a-Glance

### Dataset

- File: `data/MGNREGA_dataset_AtAGlance.csv`  
- Size: 726 rows × 34 columns  
- Granularity: One row ≈ one district’s MGNREGA summary for a given period.  

Key columns:  

- Administrative: `state_code`, `LGD_ST_Code`, `state_name`, `district_code`, `LGD_Dist_Code`, `district_name`  
- Participation: `total_no_of_job_cards_issued`, `total_no_of_workers`, `total_no_of_active_job_cards`, `total_no_of_active_workers`  
- Demographics: SC/ST worker and persondays percentages, women persondays percentage  
- Employment outcomes: `average_days_employment_per_household`, `total_no_of_HHs_completed_100_days`, `total_households_worked`, `total_individuals_worked`  
- Works: `total_no_of_works_taken_up_new_and_spillover`, `no_of_ongoing_works`, `no_of_completed_works`  
- Financials: `approved_labour_budget`, `total_expenditure`, `wages`, `material_and_skilled_wages`, `total_admin_expenditure`  

### Analysis Goals

1. Describe the dataset  
   - Shape, dtypes, head, descriptive statistics.  

2. Correlation between attributes  
   - Compute numeric correlation matrix.  
   - Inspect high correlations such as:
     - `total_expenditure` ↔ `wages`
     - `total_no_of_works_taken_up_new_and_spillover` ↔ `no_of_ongoing_works`
     - `total_households_worked` ↔ `total_individuals_worked`  

3. Conceptualize a regression problem  
   - Target: `average_days_employment_per_household`.  
   - Engineered features:
     - `active_worker_ratio` = total_no_of_active_workers / total_no_of_workers  
     - `job_card_activation_rate` = total_no_of_active_job_cards / total_no_of_job_cards_issued  
     - `wage_per_personday` = wages / persondays_of_central_liability_so_far  
     - `work_completion_rate` = no_of_completed_works / total_no_of_works_taken_up_new_and_spillover  

4. Join with Anganwadi district data  
   - Aggregate Anganwadi dataset by district name:
     - `Total_AWCs`, `Total_Children` per district.  
   - Normalize district names and perform a left join on district name.  
   - Note: Only Telangana districts match; most districts remain unmatched (high null ratio) due to coverage differences.  

5. Change attribute types  
   - Convert `state_code`, `district_code` from numeric to categorical (string).  

6. Missing values  
   - Detect nulls per column and per row.  
   - Only `LGD_Dist_Code` has a single null value.  

7. Outlier detection  
   - IQR-based detection on all numeric columns.  
   - Summarize outlier counts and percentages per column (many columns show outliers).  

### How to Run

```bash
# Assuming environment already set up
python scripts/analyze_mgnrega.py
```

Expected outputs (written to `outputs/mgnrega/`):  

- `mgnrega_correlation_matrix.csv` – full correlation matrix  
- `mgnrega_outlier_analysis.csv` – outlier summary  
- `mgnrega_regression_dataset.csv` – engineered regression dataset  
- `mgnrega_anganwadi_joined.csv` – joined district-level dataset  

***