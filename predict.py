import joblib
import pandas as pd
import numpy as np

# Đường dẫn file
MODEL_PATH = "artifacts/random_forest_model.joblib"
SCALER_PATH = "artifacts/scaler.pkl"
FEATURES_PATH = "artifacts/model_features.pkl" # File chứa tên các cột lúc train

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    return model, scaler, model_features

def preprocess_input(input_dict, scaler, model_features):
    """
    Biến đổi dữ liệu nhập vào y hệt như lúc train.
    """
    # 1. Tạo DataFrame từ input dictionary
    df = pd.DataFrame([input_dict])
    
    # 2. Binary Encoding (Thủ công cho chắc ăn)
    df['OverTime'] = 1 if df['OverTime'].iloc[0] == 'Yes' else 0
    
    # 3. One-Hot Encoding
    # Lưu ý: Nếu input là 'Single', get_dummies chỉ tạo ra cột MaritalStatus_Single
    # Nó sẽ thiếu cột MaritalStatus_Married.
    if 'MaritalStatus' in df.columns:
        df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=True)
    
    # 4. ALIGN COLUMNS (Bước QUAN TRỌNG NHẤT)
    # Reindex giúp tạo ra các cột còn thiếu (điền số 0) và sắp xếp đúng thứ tự
    df = df.reindex(columns=model_features, fill_value=0)
    
    # 5. Scaling
    # Chỉ scale các cột số thực như lúc train
    numeric_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def predict_employee(input_data):
    # 1. Load
    model, scaler, model_features = load_artifacts()
    
    # 2. Xử lý dữ liệu
    X_new = preprocess_input(input_data, scaler, model_features)
    
    # 3. Predict
    prediction = model.predict(X_new)[0]      # 0 hoặc 1
    probability = model.predict_proba(X_new)  # Xác suất [0.8, 0.2]
    
    # 4. Trả về kết quả dễ hiểu
    result = "Sẽ Nghỉ Việc (Nguy hiểm)" if prediction == 1 else "Ở lại (An toàn)"
    confidence = probability[0][prediction] * 100
    
    return result, confidence

# --- TEST THỬ ---
if __name__ == "__main__":
    # Giả lập dữ liệu nhập từ Web/App
    new_employee = {
        'Age': 25,
        'MonthlyIncome': 5500,       # Lương thấp
        'TotalWorkingYears': 2,
        'YearsAtCompany': 1,
        'OverTime': 'Yes',           # Hay làm thêm giờ
        'JobSatisfaction': 1,        # Chán việc (Mức 1)
        'MaritalStatus': 'Single'    # Độc thân
    }
    
    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    result, conf = predict_employee(new_employee)
    print(f"Dự đoán: {result}")
    print(f"Độ tin cậy: {conf:.2f}%")
