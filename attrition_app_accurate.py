"""
Employee Attrition Prediction App
Uses exact same preprocessing and encoding as the training pipeline
"""
import streamlit as st
import numpy as np
import json
import pandas as pd

# ============================================
# LOAD MODEL AND ENCODINGS
# ============================================
MODEL_PATH = r'C:\Users\ACER\Desktop\AIO\Warm_up_Session_1\AIO Conquer\archive\files\logistic_attrition_model.json'
ENCODINGS_PATH = r'C:\Users\ACER\Desktop\AIO\Warm_up_Session_1\AIO Conquer\archive\files\label_encodings.json'
@st.cache_resource
def load_model_and_encodings():
    # Load the trained model weights
    with open(MODEL_PATH, 'r') as f:
        model = json.load(f)
    
    # Load label encodings
    with open(ENCODINGS_PATH, 'r') as f:
        encodings = json.load(f)
    
    return model, encodings

model, encodings = load_model_and_encodings()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_attrition(features):
    """
    Make prediction using logistic regression
    Matches the sklearn LogisticRegression prediction logic
    """
    weights = np.array(model['weights'])
    bias = model['bias']
    
    # Calculate logit (decision function)
    logit = np.dot(features, weights) + bias
    
    # Apply sigmoid function to get probability
    probability = 1 / (1 + np.exp(-logit))
    
    return probability, logit

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<div class="main-header">👔 Employee Attrition Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Logistic Regression Model | Accuracy-Matched Preprocessing</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SIDEBAR - MODEL INFO
# ============================================
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"""
    **Model Type:** Logistic Regression  
    **Features:** {len(model['features'])}  
    **Class Weight:** Balanced  
    **Encoding:** Label Encoding (sklearn)
    
    ---
    
    **Prediction Threshold:**
    - **≥ 0.5** → Likely to Leave 🔴
    - **0.3 - 0.5** → Moderate Risk 🟡
    - **< 0.3** → Likely to Stay 🟢
    
    ---
    
    **How it works:**
    1. Input employee features
    2. Apply same encoding as training
    3. Calculate weighted sum + bias
    4. Apply sigmoid function
    5. Get probability (0-1)
    """)

# ============================================
# INPUT FORM
# ============================================
st.header("📝 Enter Employee Information")

# Create three columns for organized input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    gender = st.selectbox("Gender", options=encodings['Gender']['classes'])
    marital_status = st.selectbox("Marital Status", options=encodings['MaritalStatus']['classes'])
    distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=5, step=1)
    
    st.subheader("🎓 Education")
    education = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "1 - Below College",
            2: "2 - College",
            3: "3 - Bachelor",
            4: "4 - Master",
            5: "5 - Doctor"
        }[x],
        index=2
    )
    education_field = st.selectbox("Education Field", options=encodings['EducationField']['classes'])

with col2:
    st.subheader("💼 Job Information")
    department = st.selectbox("Department", options=encodings['Department']['classes'])
    job_role = st.selectbox("Job Role", options=encodings['JobRole']['classes'])
    job_level = st.selectbox("Job Level", options=[1, 2, 3, 4, 5], index=1)
    business_travel = st.selectbox("Business Travel", options=encodings['BusinessTravel']['classes'])
    overtime = st.selectbox("Over Time", options=encodings['OverTime']['classes'])
    
    st.subheader("⏱️ Work Experience")
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, step=1)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5, step=1)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2, step=1)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, step=1)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=2, step=1)
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2, step=1)

with col3:
    st.subheader("💰 Compensation")
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=1)
    daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=10)
    monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=15000, step=100)
    percent_salary_hike = st.slider("Percent Salary Hike (%)", min_value=10, max_value=25, value=15, step=1)
    stock_option_level = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=1)
    
    st.subheader("😊 Satisfaction & Performance")
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        index=2
    )
    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        index=2
    )
    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        index=2
    )
    work_life_balance = st.selectbox(
        "Work Life Balance",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}[x],
        index=2
    )
    job_involvement = st.selectbox(
        "Job Involvement",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        index=2
    )
    performance_rating = st.selectbox(
        "Performance Rating",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"}[x],
        index=2
    )
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2, step=1)

# ============================================
# PREDICTION BUTTON
# ============================================
st.markdown("---")
predict_button = st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)

if predict_button:
    # ============================================
    # ENCODE INPUTS (EXACT SAME AS TRAINING)
    # ============================================
    
    # Encode categorical variables using the exact same LabelEncoder mappings
    business_travel_encoded = encodings['BusinessTravel']['mapping'][business_travel]
    department_encoded = encodings['Department']['mapping'][department]
    education_field_encoded = encodings['EducationField']['mapping'][education_field]
    gender_encoded = encodings['Gender']['mapping'][gender]
    job_role_encoded = encodings['JobRole']['mapping'][job_role]
    marital_status_encoded = encodings['MaritalStatus']['mapping'][marital_status]
    overtime_encoded = encodings['OverTime']['mapping'][overtime]
    
    # Create feature array in the EXACT order as model.features
    # Features order from JSON model
    feature_dict = {
        "Age": age,
        "BusinessTravel": business_travel_encoded,
        "DailyRate": daily_rate,
        "Department": department_encoded,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field_encoded,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender_encoded,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role_encoded,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status_encoded,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime_encoded,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    }
    
    # Create feature vector in correct order
    features = np.array([feature_dict[feat] for feat in model['features']])
    
    # ============================================
    # MAKE PREDICTION
    # ============================================
    probability, logit = predict_attrition(features)
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    # Create result columns
    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
    
    with res_col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Attrition Probability",
            value=f"{probability:.2%}",
            delta=f"{probability - 0.5:.2%} from threshold" if probability >= 0.5 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with res_col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Decision Score (Logit)",
            value=f"{logit:.4f}",
            help="Raw output before sigmoid transformation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with res_col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Determine risk level and prediction
        if probability >= 0.5:
            prediction = "LIKELY TO LEAVE"
            risk_level = "🔴 HIGH RISK"
            color = "red"
        elif probability >= 0.3:
            prediction = "MODERATE RISK"
            risk_level = "🟡 MODERATE RISK"
            color = "orange"
        else:
            prediction = "LIKELY TO STAY"
            risk_level = "🟢 LOW RISK"
            color = "green"
        
        st.metric(label="Prediction", value=prediction)
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Probability visualization
    st.markdown("### 📈 Probability Breakdown")
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.markdown("**Stay Probability**")
        st.progress(1 - probability)
        st.caption(f"{(1-probability):.2%}")
    
    with prob_col2:
        st.markdown("**Leave Probability**")
        st.progress(probability)
        st.caption(f"{probability:.2%}")
    
    # ============================================
    # FEATURE CONTRIBUTIONS (Top 5)
    # ============================================
    st.markdown("---")
    st.subheader("🎯 Top 5 Contributing Factors")
    
    # Calculate feature contributions
    contributions = features * np.array(model['weights'])
    feature_contributions = pd.DataFrame({
        'Feature': model['features'],
        'Value': features,
        'Weight': model['weights'],
        'Contribution': contributions
    })
    
    # Get top 5 positive and negative contributors
    top_positive = feature_contributions.nlargest(5, 'Contribution')
    top_negative = feature_contributions.nsmallest(5, 'Contribution')
    
    contrib_col1, contrib_col2 = st.columns(2)
    
    with contrib_col1:
        st.markdown("**Increasing Attrition Risk** 🔺")
        for _, row in top_positive.iterrows():
            st.markdown(f"- **{row['Feature']}**: {row['Contribution']:.4f}")
    
    with contrib_col2:
        st.markdown("**Decreasing Attrition Risk** 🔻")
        for _, row in top_negative.iterrows():
            st.markdown(f"- **{row['Feature']}**: {row['Contribution']:.4f}")
    
    # ============================================
    # RECOMMENDATIONS
    # ============================================
    st.markdown("---")
    st.subheader("💡 Actionable Recommendations")
    
    if probability >= 0.5:
        st.error("""
        **⚠️ HIGH ATTRITION RISK - IMMEDIATE ACTION REQUIRED**
        
        **Priority Actions:**
        1. 🗣️ **Schedule urgent one-on-one meeting** - Understand concerns and challenges
        2. 💰 **Review compensation package** - Benchmark against market rates
        3. 📈 **Create development plan** - Discuss career growth and promotion opportunities
        4. ⚖️ **Assess workload** - Consider overtime, work-life balance adjustments
        5. 🎯 **Set clear goals** - Ensure employee sees path forward in organization
        
        **Additional Considerations:**
        - Review job satisfaction and work environment factors
        - Consider role adjustment or internal transfer opportunities
        - Ensure adequate recognition and appreciation
        - Evaluate manager-employee relationship
        """)
    elif probability >= 0.3:
        st.warning("""
        **⚠️ MODERATE ATTRITION RISK - PROACTIVE ENGAGEMENT NEEDED**
        
        **Recommended Actions:**
        1. 📅 **Regular check-ins** - Schedule monthly 1-on-1s to monitor satisfaction
        2. 🏆 **Recognition program** - Ensure contributions are acknowledged
        3. 📚 **Training opportunities** - Invest in skill development
        4. 🎯 **Career discussions** - Clarify growth path and timeline
        5. 🤝 **Team engagement** - Foster stronger workplace relationships
        
        **Monitor These Areas:**
        - Job satisfaction trends
        - Work-life balance indicators
        - Performance and engagement levels
        """)
    else:
        st.success("""
        **✅ LOW ATTRITION RISK - MAINTAIN POSITIVE MOMENTUM**
        
        **Best Practices:**
        1. 🌟 **Continue positive practices** - Keep doing what's working
        2. 🎯 **Provide challenges** - Offer new projects to maintain engagement
        3. 🏆 **Recognize achievements** - Regular acknowledgment of contributions
        4. 📈 **Support growth** - Continue investing in development
        5. 💬 **Stay connected** - Maintain open communication channels
        
        **Ongoing Focus:**
        - Keep employee engaged with meaningful work
        - Ensure competitive compensation
        - Support career advancement goals
        """)
    
    # ============================================
    # DETAILED INPUT SUMMARY
    # ============================================
    with st.expander("📋 View Input Summary & Encoded Values"):
        summary_df = pd.DataFrame({
            'Feature': model['features'],
            'Input Value': features,
            'Weight': model['weights'],
            'Contribution': contributions
        })
        st.dataframe(summary_df, use_container_width=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
    <strong>Employee Attrition Prediction System</strong><br>
    Logistic Regression Model with Balanced Class Weights<br>
    Preprocessing: Label Encoding (sklearn.preprocessing.LabelEncoder)<br>
    <em>This prediction is a statistical estimate. Use as one input in retention decisions.</em>
    </small>
</div>
""", unsafe_allow_html=True)
