# Triển khai ứng dụng mô hình Machine Learning bằng Streamlit

## 1. Giới thiệu về Deploy mô hình Machine Learning

Sau khi hoàn thành quá trình xử lý dữ liệu và xây dựng mô hình Machine
Learning, bước tiếp theo là triển khai (deploy) mô hình thành một ứng
dụng thực tế để người dùng có thể sử dụng. Việc deploy giúp mô hình
không chỉ dừng lại ở mức thử nghiệm mà có thể áp dụng vào thực tế, hỗ
trợ dự đoán hoặc ra quyết định.

Trong dự án này, nhóm sử dụng Streamlit để triển khai mô hình. Streamlit
là một framework Python cho phép xây dựng giao diện web đơn giản và
nhanh chóng dành cho các ứng dụng Data Science và Machine Learning.

------------------------------------------------------------------------

## 2. Lý do lựa chọn Streamlit

Streamlit được lựa chọn do các ưu điểm sau:

-   **Dễ sử dụng:** Không yêu cầu kiến thức chuyên sâu về Front-end.
-   **Tích hợp tốt với Python:** Phù hợp với các mô hình Machine
    Learning đã được xây dựng bằng Python.
-   **Triển khai nhanh:** Chỉ cần viết vài dòng lệnh là có thể tạo giao
    diện web.
-   **Hỗ trợ visualization:** Có thể hiển thị biểu đồ và kết quả phân
    tích trực quan.

------------------------------------------------------------------------

## 3. Quy trình triển khai ứng dụng Streamlit

### 3.1 Cài đặt các thư viện cần thiết

``` bash
pip install streamlit
pip install scikit-learn
pip install pandas
pip install joblib
```

------------------------------------------------------------------------

### 3.2 Xây dựng UI

#### a. Phần Input của người dùng

![UI nhập liệu của người dùng](C:\Users\ACER\Desktop\AIO\Warm_up_Session_1\AIO Conquer\archive\deploy_streamlit ver00.00.01\Blog\Feature\User_input.jpeg.png)


``` python
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)

with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3,
        help="1: Low, 2: Medium, 3: High, 4: Very High"
    )

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)
```

------------------------------------------------------------------------

#### b. UI hiển thị kết quả dự đoán

![UI kết quả 2 mô hình](C:\Users\ACER\Desktop\AIO\Warm_up_Session_1\AIO Conquer\archive\deploy_streamlit ver00.00.01\Blog\Feature\Output.jpg)

Về phần này, UI sẽ cho ta thấy sự khác biệt của mô hình khi dự đoán hai kết quả khác nhau

``` python
col1, col2 = st.columns(2)

with col1:
    st.subheader("Random Forest")
    if rf_prediction == 1:
        st.error(f"{rf_result}")
    else:
        st.success(f"### {rf_result}")
    st.markdown(f"**Confidence:** {rf_conf:.2f}%")

    st.markdown("#### Probabilities")
    st.metric("Stay", f"{rf_probabilities[0]*100:.2f}%")
    st.metric("Leave", f"{rf_probabilities[1]*100:.2f}%")

    st.progress(float(rf_probabilities[1]))
    leave_prob_rf = rf_probabilities[1] * 100
    if leave_prob_rf < 30:
        st.markdown(f":green[Low Risk: {leave_prob_rf:.1f}%]")
    elif leave_prob_rf < 60:
        st.markdown(f":orange[Medium Risk: {leave_prob_rf:.1f}%]")
    else:
        st.markdown(f":red[High Risk: {leave_prob_rf:.1f}%]")

with col2:
    st.subheader("Logistic Regression")
    if lr_prediction == 1:
        st.error(f"### {lr_result}")
    else:
        st.success(f"### {lr_result}")
    st.markdown(f"**Confidence:** {lr_conf:.2f}%")

    st.markdown("#### Probabilities")
    st.metric("Stay", f"{lr_probabilities[0]*100:.2f}%")
    st.metric("Leave", f"{lr_probabilities[1]*100:.2f}%")

    st.progress(float(lr_probabilities[1]))
```

------------------------------------------------------------------------

## 4. Hạn chế và hướng phát triển

### Hạn chế

-   Giao diện Streamlit còn đơn giản
-   Khả năng xử lý dữ liệu lớn còn hạn chế
-   Chưa tối ưu hiệu suất mô hình

### Hướng phát triển

-   Tối ưu giao diện người dùng
-   Tích hợp nhiều mô hình dự đoán
-   Triển khai trên server để phục vụ nhiều người dùng đồng thời
-   Triển khai trên cloud để cho mọi người khác sử dụng thử mô hình
    (Hugging Face)

------------------------------------------------------------------------

## 8. Kết luận

Việc sử dụng Streamlit giúp quá trình triển khai mô hình Machine
Learning trở nên nhanh chóng và hiệu quả. Công cụ này giúp kết nối giữa
mô hình và người dùng cuối, góp phần đưa các mô hình AI vào ứng dụng
thực tế.
