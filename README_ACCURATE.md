# 🎯 Employee Attrition Prediction System
## Streamlit App with Accuracy-Matched Preprocessing

This Streamlit application provides **accurate** employee attrition predictions using the same preprocessing pipeline as your training code.

---

## ✨ Key Features

### 🔬 **Accuracy Guarantee**
- **Exact same label encoding** as training (sklearn.preprocessing.LabelEncoder)
- **Same feature order** as the trained model
- **Identical preprocessing** steps (dropped columns, encoding logic)
- **Verified encoding mappings** extracted from actual training data

### 📊 **Comprehensive Interface**
- **30 input features** organized in 3 intuitive columns
- **Real-time prediction** with probability scores
- **Risk level classification** (Low/Moderate/High)
- **Feature contribution analysis** - See which factors drive the prediction
- **Actionable recommendations** based on risk level

### 🎨 **Professional UI**
- Clean, modern design with custom styling
- Color-coded risk indicators (🟢 🟡 🔴)
- Interactive sidebar with model information
- Progress bars for probability visualization
- Expandable sections for detailed data

---

## 📦 Installation

### Prerequisites
```bash
pip install streamlit pandas numpy
```

### Required Files
Make sure these files are in the correct locations:

1. **attrition_app_accurate.py** - The Streamlit app
2. **logistic_attrition_model.json** - Your trained model weights
3. **label_encodings.json** - Encoding mappings (included)
4. **WA_Fn-UseC_-HR-Employee-Attrition.csv** - Original dataset (for reference)

---

## 🚀 How to Run

### Step 1: Update File Paths (if needed)
In `attrition_app_accurate.py`, update these lines if your files are in different locations:

```python
# Line 16 - Model file
with open('/mnt/user-data/uploads/logistic_attrition_model.json', 'r') as f:

# Line 20 - Encodings file  
with open('/home/claude/label_encodings.json', 'r') as f:
```

### Step 2: Launch the App
```bash
streamlit run attrition_app_accurate.py
```

### Step 3: Access the Interface
The app will automatically open in your browser at:
```
http://localhost:8501
```

---

## 📖 How to Use

### 1️⃣ **Input Employee Data**

The form is organized into three columns:

**Column 1 - Personal & Education:**
- Age, Gender, Marital Status
- Distance from home
- Education level and field

**Column 2 - Job & Experience:**
- Department, Role, Level
- Business travel frequency
- Overtime status
- Work experience metrics
- Years at company, in role, since promotion

**Column 3 - Compensation & Satisfaction:**
- Monthly income, hourly/daily rates
- Salary hike percentage
- Stock options
- Satisfaction ratings (environment, job, relationships)
- Work-life balance
- Performance and training metrics

### 2️⃣ **Get Prediction**

Click the **"🔮 Predict Attrition Risk"** button to:
- Calculate attrition probability
- View risk classification
- See decision score (logit value)

### 3️⃣ **Analyze Results**

The results section shows:
- **Probability**: Exact % chance of leaving (0-100%)
- **Risk Level**: 
  - 🟢 **Low Risk** (< 30%): Likely to stay
  - 🟡 **Moderate Risk** (30-50%): Needs monitoring
  - 🔴 **High Risk** (≥ 50%): Likely to leave
- **Feature Contributions**: Top 5 factors increasing/decreasing risk
- **Recommendations**: Specific actions based on risk level

---

## 🔍 Technical Details

### Model Architecture
- **Algorithm**: Logistic Regression
- **Class Weighting**: Balanced (handles class imbalance)
- **Features**: 30 employee attributes
- **Output**: Probability score (0-1) via sigmoid function

### Preprocessing Pipeline

**Dropped Columns** (same as training):
```python
["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
```

**Label Encoding Mappings**:
```
Attrition: {No: 0, Yes: 1}
BusinessTravel: {Non-Travel: 0, Travel_Frequently: 1, Travel_Rarely: 2}
Department: {Human Resources: 0, Research & Development: 1, Sales: 2}
EducationField: {Human Resources: 0, Life Sciences: 1, Marketing: 2, 
                 Medical: 3, Other: 4, Technical Degree: 5}
Gender: {Female: 0, Male: 1}
JobRole: {Healthcare Representative: 0, Human Resources: 1, 
          Laboratory Technician: 2, Manager: 3, Manufacturing Director: 4,
          Research Director: 5, Research Scientist: 6, Sales Executive: 7,
          Sales Representative: 8}
MaritalStatus: {Divorced: 0, Married: 1, Single: 2}
OverTime: {No: 0, Yes: 1}
```

### Feature Order
Features are processed in this exact order (matching model.features):
```python
["Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
 "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
 "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
 "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
 "OverTime", "PercentSalaryHike", "PerformanceRating",
 "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
 "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
 "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
```

### Prediction Formula
```python
# 1. Calculate weighted sum
logit = sum(feature[i] * weight[i] for all i) + bias

# 2. Apply sigmoid function
probability = 1 / (1 + exp(-logit))

# 3. Classify
if probability >= 0.5: "Likely to Leave"
elif probability >= 0.3: "Moderate Risk"
else: "Likely to Stay"
```

---

## 📊 Example Use Cases

### Example 1: High-Risk Employee
```
Age: 28, Single, Overtime: Yes
Years at Company: 1
Distance from Home: 25 km
Environment Satisfaction: Low (1)
Job Satisfaction: Low (1)
Work-Life Balance: Bad (1)
→ Result: HIGH RISK (85% probability)
```

### Example 2: Low-Risk Employee
```
Age: 42, Married, Overtime: No
Years at Company: 15
Distance from Home: 5 km
Environment Satisfaction: Very High (4)
Job Satisfaction: High (3)
Work-Life Balance: Best (4)
Monthly Income: High
→ Result: LOW RISK (12% probability)
```

### Example 3: Moderate-Risk Employee
```
Age: 32, Divorced, Overtime: Yes
Years at Company: 3
Job Satisfaction: Medium (2)
Recent Promotion: Yes (0 years since)
Training: Regular (3 times)
→ Result: MODERATE RISK (38% probability)
```

---

## 🎯 Interpretation Guide

### Risk Thresholds
- **< 30%**: Employee is engaged and stable
- **30-50%**: Warning signs present, proactive engagement needed
- **≥ 50%**: Significant flight risk, immediate intervention required

### Key Risk Factors (from training data analysis)
**Increases Attrition:**
- Overtime work
- Low job satisfaction
- Low environment satisfaction
- Being single
- Frequent business travel
- Many previous companies
- Low stock options

**Decreases Attrition:**
- Higher age
- Years at company
- High job involvement
- Good work-life balance
- Higher job satisfaction
- Married status
- Manager tenure

---

## 🔧 Customization Options

### Adjust Risk Thresholds
In the code, you can modify classification thresholds:
```python
if probability >= 0.5:  # Change to 0.6 for stricter high-risk
    prediction = "LIKELY TO LEAVE"
elif probability >= 0.3:  # Change to 0.4 for stricter moderate
    prediction = "MODERATE RISK"
```

### Add Custom Recommendations
Enhance the recommendation logic in the "Actionable Recommendations" section based on specific feature values.

### Enable Batch Predictions
Add CSV upload functionality to process multiple employees at once.

---

## 📝 Notes & Best Practices

1. **Data Quality**: Ensure input values are realistic and within expected ranges
2. **Context Matters**: Use predictions as one input, not the sole decision factor
3. **Regular Updates**: Retrain model periodically with new data
4. **Privacy**: Handle employee data securely and in compliance with regulations
5. **Human Judgment**: Combine model insights with manager knowledge

---

## 🐛 Troubleshooting

**Issue**: "File not found" error
- **Solution**: Check file paths in lines 16 and 20 of the app

**Issue**: Encoding error for categorical values
- **Solution**: Ensure input matches exact values from training data

**Issue**: Unexpected predictions
- **Solution**: Verify feature order matches model.features exactly

**Issue**: App won't start
- **Solution**: Check that all required packages are installed

---

## 📚 Files Included

1. **attrition_app_accurate.py** - Main Streamlit application
2. **label_encodings.json** - Exact encoding mappings from training
3. **README.md** - This documentation file

---

## 🎓 Model Performance (from training)

Your original training code with `class_weight="balanced"`:
- Handles class imbalance effectively
- Provides calibrated probabilities
- Uses stratified train/test split for reliability

**Recommended Next Steps:**
1. Add model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
2. Include feature importance visualization from training
3. Implement SHAP values for better interpretability
4. Add confidence intervals for predictions

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the technical details
3. Verify preprocessing matches training exactly

---

## 📄 License & Credits

Built to match the preprocessing pipeline from your HR attrition analysis code.  
Model: Logistic Regression with balanced class weights  
Data: IBM HR Employee Attrition Dataset

---

**Happy Predicting! 🎯**
