# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install streamlit pandas numpy
```

### Step 2: Run the App
```bash
streamlit run attrition_app_accurate.py
```

### Step 3: Open Browser
The app automatically opens at: `http://localhost:8501`

---

## 📝 Quick Test

Try these scenarios to see the model in action:

### 🔴 High Risk Employee
- Age: 28
- Marital Status: Single
- Over Time: Yes
- Years at Company: 1
- Job Satisfaction: 1 (Low)
- Environment Satisfaction: 1 (Low)
- Work Life Balance: 1 (Bad)
- Distance From Home: 25 km

**Expected Result:** ~70-85% attrition probability

---

### 🟢 Low Risk Employee
- Age: 45
- Marital Status: Married
- Over Time: No
- Years at Company: 15
- Job Satisfaction: 4 (Very High)
- Environment Satisfaction: 4 (Very High)
- Work Life Balance: 4 (Best)
- Distance From Home: 5 km
- Monthly Income: 15000

**Expected Result:** ~5-15% attrition probability

---

## ✅ Verification

Run the verification script to confirm accuracy:
```bash
python verify_accuracy.py
```

This will show:
- Sample predictions on first 10 employees
- Overall model accuracy (~70%)
- Feature order verification
- Detailed calculation example

**Verified Results:**
- ✓ Feature encoding matches training
- ✓ Feature order matches model
- ✓ Predictions are accurate
- ✓ Model accuracy: ~70%

---

## 📊 What You'll See

### Main Interface
1. **Sidebar** - Model info and instructions
2. **Input Form** - 30 employee features in 3 columns
3. **Predict Button** - Click to get results

### Results Display
1. **Attrition Probability** - Exact percentage (0-100%)
2. **Decision Score** - Raw logit value before sigmoid
3. **Prediction** - Stay/Moderate/Leave classification
4. **Risk Level** - 🟢 Low / 🟡 Moderate / 🔴 High
5. **Probability Breakdown** - Visual progress bars
6. **Top 5 Factors** - Features increasing/decreasing risk
7. **Recommendations** - Specific actions based on risk

---

## 🎯 Key Features

- ✅ **Accuracy-Matched** - Same preprocessing as training
- ✅ **Real-time** - Instant predictions
- ✅ **Interactive** - Easy-to-use interface
- ✅ **Insightful** - Shows feature contributions
- ✅ **Actionable** - Provides recommendations
- ✅ **Professional** - Clean, modern design

---

## 💡 Pro Tips

1. **Test with extremes** - Try min/max values to understand model behavior
2. **Check contributions** - See which factors drive each prediction
3. **Compare scenarios** - Test similar employees with one changed factor
4. **Use recommendations** - Follow suggested actions for each risk level
5. **Verify inputs** - Ensure realistic values for accurate predictions

---

## 🔧 File Locations

Make sure these files are accessible:

```
project/
├── attrition_app_accurate.py     # Main app
├── label_encodings.json           # Encoding mappings
├── logistic_attrition_model.json  # Model weights
└── verify_accuracy.py             # Verification script
```

---

## 📞 Need Help?

1. **App won't start?** - Check dependencies are installed
2. **File not found?** - Update paths in lines 16 & 20 of the app
3. **Unexpected results?** - Run verify_accuracy.py to check
4. **Questions?** - See README_ACCURATE.md for details

---

**Ready to predict! 🎯**
