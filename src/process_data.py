import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CẤU HÌNH ---
# Đường dẫn file gốc và nơi lưu kết quả
RAW_DATA_PATH = "D:\AIO\AIO_The Liems\M01\Project\TheLiems_Conquer\data\raw\WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DIR = "D:\AIO\AIO_The Liems\M01\Project\TheLiems_Conquer\data\processed"     # Nơi chứa data sạch
ARTIFACTS_DIR = "D:\AIO\AIO_The Liems\M01\Project\TheLiems_Conquer\data\artifacts"     # Nơi chứa scaler/encoder

# Tạo thư mục nếu chưa có
# os.makedirs(PROCESSED_DIR, exist_ok=True)
# os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def processing_pipeline():
    print("🚀 BẮT ĐẦU QUY TRÌNH XỬ LÝ DỮ LIỆU...")

    # 1. LOAD DỮ LIỆU
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {RAW_DATA_PATH}")
        return
        
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✓ Đã tải dữ liệu gốc: {df.shape}")

    # 2. LÀM SẠCH (CLEANING)
    # Loại bỏ các cột không có giá trị thông tin
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # (Tùy chọn) Đổi tên cột nếu cần thiết cho chuẩn hóa
    # df.columns = [col.lower() for col in df.columns] 

    # 3. XỬ LÝ TARGET (Mục tiêu dự đoán)
    # Chuyển Attrition từ 'Yes'/'No' sang 1/0
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # ==> LƯU BẢN CLEAN SƠ BỘ
    clean_path = os.path.join(PROCESSED_DIR, 'hr_cleaned_readable.csv')
    df.to_csv(clean_path, index=False)
    print(f"✓ Đã lưu file sạch (dạng đọc được): {clean_path}")

    # 4. FEATURE ENGINEERING
    # Tách biến
    target_col = 'Attrition'
    categorical_cols = [col for col in df.select_dtypes(include='object').columns if col != target_col]
    numerical_cols = [col for col in df.select_dtypes(exclude='object').columns if col != target_col]

    # A. Encoding (Chữ -> Số)
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Lưu lại bộ mã hóa để dùng lại sau này
    joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl'))
    print("✓ Đã lưu Label Encoders")

    # B. Chia Train/Test (Split)
    # Tách X (Đặc trưng) và y (Nhãn)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Chia 80% Train - 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # C. Scaling (Chuẩn hóa số liệu)
    # Chỉ fit trên tập TRAIN để tránh rò rỉ dữ liệu (data leakage) sang tập Test
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Lưu lại bộ Scaler
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    print("✓ Đã lưu Scaler")

    # 5. LƯU KẾT QUẢ CUỐI CÙNG (FINAL SAVE)
    # Lưu dưới dạng CSV để dễ kiểm tra
    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

    print(f"\n✅ HOÀN TẤT! Dữ liệu đã sẵn sàng tại thư mục: {PROCESSED_DIR}")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test:  {X_test.shape}")

if __name__ == "__main__":
    processing_pipeline()