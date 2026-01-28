import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from utils.logger_config import logger

# Cấu hình đường dẫn
ARTIFACTS_DIR = "./artifacts"
PROCESSED_DIR = "./data/processed"

def create_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def select_important_features(df):
    selected_columns = [
        'Attrition',           # Target
        'OverTime',            # Feature 1 (Yes/No)
        'MonthlyIncome',       # Feature 2 (Int)
        'Age',                 # Feature 3 (Int)
        'TotalWorkingYears',   # Feature 4 (Int)
        'YearsAtCompany',      # Feature 5 (Int)
        'JobSatisfaction',     # Feature 6 (1-4)
        'MaritalStatus'        # Feature 7 (Single/Married/Divorced)
    ]
    
    # Kiểm tra xem đủ cột không
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError("Dataset is missing required columns")
        
    df_reduce = df[selected_columns].copy()
    logger.info(f"Reduced dataset to {len(selected_columns)} columns.")
    
    # Lưu danh sách cột để sau này App dùng
    cols_path = os.path.join(ARTIFACTS_DIR, 'model_columns.pkl')
    # Lưu danh sách cột features (bỏ Attrition)
    feature_cols = [c for c in selected_columns if c != 'Attrition']
    
    # Lưu ý: Sau này One-Hot Encoding thì số lượng cột sẽ tăng lên (MaritalStatus -> 2 cột)
    # Nên ta sẽ lưu lại sau bước Encoding
    return df_reduce

def encode_data(df):
    """
    Xử lý mã hóa: Binary (Yes/No) và One-Hot (MaritalStatus).
    """
    df = df.copy()
    
    # 1. Binary Encoding (Yes/No -> 1/0)
    # Áp dụng cho Attrition và OverTime
    binary_cols = ['Attrition', 'OverTime']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 2. One-Hot Encoding cho MaritalStatus
    # drop_first=True để tránh đa cộng tuyến (3 giá trị -> 2 cột)
    if 'MaritalStatus' in df.columns:
        df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=True)
        # Vì get_dummies tạo cột boolean (True/False), chuyển về int (1/0)
        new_cols = [c for c in df.columns if 'MaritalStatus' in c]
        for c in new_cols:
            df[c] = df[c].astype(int)
            
    logger.success("Data encoding (Binary & One-Hot) completed.")
    return df

def process_and_split(df):
    """
    Chia train/test và Scale dữ liệu.
    """
    target_col = 'Attrition'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Lưu danh sách cột cuối cùng (Sau khi One-Hot) để App dùng sắp xếp đúng thứ tự
    final_features = X.columns.tolist()
    joblib.dump(final_features, os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
    logger.info(f"Saved feature list: {final_features}")

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Scaling (Chỉ scale cột số thực, tránh cột 0/1)
    # Các cột cần scale: Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany
    # JobSatisfaction là thứ tự (1-4) có thể để nguyên hoặc scale cũng được
    numeric_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Lưu Scaler
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    logger.success("Scaling completed and Scaler saved.")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Cân bằng dữ liệu bằng SMOTE (áp dụng trên tập Train).
    """
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE: {y_train_resampled.value_counts().to_dict()}")
    logger.success("SMOTE applied successfully.")
    
    return X_train_resampled, y_train_resampled

def save_data(X_train, X_test, y_train, y_test):
    """Lưu kết quả cuối cùng."""
    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)
    logger.success(f"Processed data saved to {PROCESSED_DIR}")