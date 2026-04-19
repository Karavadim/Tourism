
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset

# --- Configuration from environment variables (for GitHub Actions) ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN environment variable not set. Please set it.")
    exit(1)

login(token=HF_TOKEN, add_to_git_credential=False)

HF_USERNAME = os.environ.get("HF_USERNAME", "Karavadi")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USERNAME}/tourism-package-dataset")

MASTER_DIR = "tourism-prediction_mlops"
DATA_DIR   = os.path.join(MASTER_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("--- Running data preparation script ---")

# 1. Download raw data from Hugging Face
csv_filename = "tourism.csv"
local_raw_data_path = os.path.join(DATA_DIR, csv_filename)

try:
    # Use a dummy path for hf_hub_download, then move to desired local_raw_data_path
    downloaded_file_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=csv_filename,
        repo_type="dataset"
    )
    # Move downloaded file to the expected DATA_DIR
    import shutil
    shutil.move(downloaded_file_path, local_raw_data_path)
    print(f"✅ Raw dataset '{csv_filename}' downloaded to {local_raw_data_path}!")
except Exception as e:
    print(f"❌ Error downloading '{csv_filename}' from Hugging Face: {e}")
    exit(1)

# 2. Load and clean the data
try:
    data = pd.read_csv(local_raw_data_path)
    print(f"✅ Loaded tourism.csv from {local_raw_data_path}")
except FileNotFoundError:
    print(f"❌ File not found: {local_raw_data_path}")
    exit(1)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Club 'Unmarried' with 'Single' in 'MaritalStatus' column
if 'MaritalStatus' in data.columns:
    data['MaritalStatus'] = data['MaritalStatus'].replace({'Unmarried': 'Single'})
    print("✅ Clubbed 'Unmarried' with 'Single' in the 'MaritalStatus' column.")

# Correct the 'Fe Male' entry to 'Female' in the 'Gender' column
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].replace({'Fe Male': 'Female'})
    print("✅ Corrected 'Fe Male' to 'Female' in the 'Gender' column.")

df_clean = data.copy()

# Drop identifier columns — not predictive features
df_clean = df_clean.drop(columns=['CustomerID', 'Unnamed: 0'], errors='ignore')
print("✅ Dropped CustomerID and Unnamed: 0 (non-predictive features).")

# IQR-based outlier capping for numerical columns
outlier_cols = ['Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips']
print("✅ Applying IQR-based outlier capping for numerical columns.")
for col in outlier_cols:
    if col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Ensure lower bound is not less than zero for applicable columns
        if col in ['Age', 'DurationOfPitch', 'NumberOfTrips']:
            lower = max(0, lower)

        df_clean[col] = df_clean[col].clip(lower, upper)
        print(f"  → {col}: Capped outliers (bounds: [{lower:.1f}, {upper:.1f}]).")

# 3. Split the cleaned dataset into training and testing sets
X = df_clean.drop('ProdTaken', axis=1)
y = df_clean['ProdTaken']

categorical_cols = X.select_dtypes(include='object').columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data splitting complete!")

# 4. Save the processed datasets locally
X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)

print(f"✅ Processed datasets saved to {PROCESSED_DATA_DIR}/")

# 5. Upload the resulting train and test datasets back to the Hugging Face data space
api = HfApi(token=HF_TOKEN)

files_to_upload = {
    'X_train.csv': os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'),
    'y_train.csv': os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'),
    'X_test.csv': os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'),
    'y_test.csv': os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')
}

print(f"Uploading processed datasets to Hugging Face dataset: {DATASET_REPO}/")

for filename, local_path in files_to_upload.items():
    try:
        remote_path = f"data/processed/{filename}"

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN # Ensure HF_TOKEN is defined.
        )
        print(f"✅ Successfully uploaded {filename} to {DATASET_REPO}/{remote_path}")
    except Exception as e:
        print(f"❌ Error uploading {filename}: {e}")

print("--- Data preparation script finished ---")
