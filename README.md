# 🚀 Employee Attrition Prediction (Dự đoán Nghỉ việc)
Dự án Machine Learning nhằm dự đoán nguy cơ nghỉ việc của nhân viên. Bằng cách phân tích các yếu tố như lương, tuổi tác, và áp lực công việc, hệ thống giúp bộ phận HR đưa ra các biện pháp giữ chân nhân tài kịp thời.

## 📂 Cấu trúc thư mục
Plaintext

CONQUER/
├── artifacts/              # Lưu trữ Model (.joblib) & Scaler (.pkl) sau khi huấn luyện
├── data/                   # Dữ liệu đầu vào (raw) và dữ liệu sau xử lý (processed)
├── data_processing/        # Module xử lý: Clean, Encode, Split, SMOTE
├── train/                  # Module huấn luyện: Logistic Regression, Random Forest
├── utils/                  # Tiện ích bổ trợ: Logger, cấu hình hệ thống
├── main.py                 # 🟢 File chính để CHẠY TRAIN MODEL (Pipeline)
├── predict.py              # 🔵 File dùng để DỰ ĐOÁN (Inference)
└── requirements.txt        # Danh sách thư viện cần cài đặt
## 🛠️ Cài đặt môi trường
1. Yêu cầu: Đã cài đặt Python 3.9+.

2. Cài đặt thư viện: Mở terminal tại thư mục dự án và chạy:

Bash

pip install -r requirements.txt
Danh sách gồm: pandas, scikit-learn, imbalanced-learn, joblib, loguru, numpy.

## 🚀 Hướng dẫn sử dụng
1️⃣ Huấn luyện Mô hình (Training)
Để xử lý dữ liệu từ file gốc, cân bằng dữ liệu và huấn luyện model, hãy chạy:

Bash

python main.py
Kết quả: Sau khi chạy xong, các file model.joblib và scaler.pkl sẽ tự động được tạo/cập nhật trong thư mục artifacts/.

Log: Tiến trình sẽ được hiển thị chi tiết qua console nhờ loguru.

2️⃣ Dự đoán nhân viên mới (Prediction)
Để kiểm tra xem một nhân viên cụ thể có khả năng nghỉ việc hay không:

Bash

python predict.py
Lưu ý: Bạn có thể mở file predict.py và thay đổi các thông số đầu vào trong biến sample_employee để thử nghiệm các kịch bản khác nhau.

## 📊 Quy trình xử lý dữ liệu (Logic)
Dự án tập trung vào 7 tính năng (Features) quan trọng nhất ảnh hưởng đến quyết định nghỉ việc: Age, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, v.v.

Các bước xử lý (Preprocessing):
One-Hot Encoding: Chuyển đổi các thông tin dạng chữ (như MaritalStatus) sang dạng số (0 và 1) bằng pd.get_dummies.

SMOTE (Synthetic Minority Over-sampling Technique): Tự động tạo thêm dữ liệu giả lập cho nhóm nhân viên nghỉ việc để tránh tình trạng model bị lệch (mất cân bằng dữ liệu).

StandardScaler: Chuẩn hóa các thang đo (ví dụ: đưa Lương và Tuổi về cùng một khoảng giá trị) giúp mô hình tính toán chính xác hơn.

Mô hình sử dụng:
1. Logistic Regression: Dùng làm base-line, giúp hiểu rõ mức độ ảnh hưởng của từng yếu tố.

2. Random Forest: Cung cấp độ chính xác cao hơn, xử lý tốt các mối quan hệ phi tuyến tính phức tạp giữa các đặc trưng.

## 📝 Ghi chú
Mọi thông tin trong quá trình chạy đều được lưu log để dễ dàng truy vết lỗi.

File .joblib trong artifacts/ là file nhị phân, không nên chỉnh sửa trực tiếp.