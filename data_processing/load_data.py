import os
import shutil
import pandas as pd
from utils.logger_config import logger
import kagglehub

def load_data():
    current_dir = os.getcwd() # xác định dynamic động thư mục làm việc hiện tại
    target_dir = os.path.join(current_dir, "data")
    file_name = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    final_path = os.path.join(target_dir, file_name)

    # 1. Nếu file chưa có ở ....\data -> Tải từ Kaggle và Copy về
    if not os.path.exists(final_path):
        logger.info(f"File not found at {final_path}. Downloading...")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(target_dir, exist_ok=True)
        
        # Tải về (kagglehub khi tải về được lưu trong thư mục temp trong ổ C)
        temp_path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        source_file = os.path.join(temp_path, file_name)
        
        # Copy sang dynamic folder ....\data
        shutil.copy(source_file, final_path)
        logger.success(f"Downloaded and saved to: {final_path}")
    else:
        logger.info(f"Found existing file at: {final_path}")

    # 2. Load data từ đúng vị trí mong muốn
    df = pd.read_csv(final_path)
    logger.success(f"Loaded dataframe. Shape: {df.shape}")
    
    return df
