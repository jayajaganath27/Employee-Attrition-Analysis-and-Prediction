import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="HR Intelligence System", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_resources():
    # 1. Load the Models
    attr_model = joblib.load("gradient_boosting_best.pkl")
    perf_model = joblib.load("linear_regression_model.pkl")
    
    # 2. Load the Scaler
    scaler = joblib.load("linear_regression_scaler.pkl")
    
    # 3. Load Feature Lists (The "DNA" of your models)
    # We extract the names to ensure matching order
    attr_feat_data = joblib.load("gb_feature_importance.pkl") 
    if isinstance(attr_feat_data, pd.DataFrame):
        attr_features = attr_feat_data['Feature'].tolist()
        attr_importance_df = attr_feat_data.set_index('Feature').sort_values(by='Importance', ascending=False)
    else:
        attr_features = list(attr_feat_data)
        attr_importance_df = None

    perf_features = list(joblib.load("linear_regression_features.pkl"))
    
    return attr_model, perf_model, scaler, attr_features, perf_features, attr_importance_df

# Initialize Assets
try:
    attr_model, perf_model, scaler, attr_features, perf_features, attr_importance_df = load_resources()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

st.title("📊 HR Analytics Prediction Portal")

# --- INPUT SECTION ---
st.header("Step 1: Employee Input Data")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Basic Info")
    age = st.slider("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    st.subheader("💼 Career History")
    years_at_co = st.slider("Years at Company", 0, 40, 5)
    total_work_years = st.slider("Total Working Years", 0, 40, 10)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ])
    overtime = st.radio("Works Overtime?", ["Yes", "No"], horizontal=True)

with col3:
    st.subheader("🎯 Performance & Engagement")
    job_sat = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
    env_sat = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3)
    job_inv = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3)
    hike = st.slider("Salary Hike (%)", 10, 30, 15)
    dist_home = st.slider("Distance From Home (km)", 1, 30, 5)

# --- PREPROCESSING ENGINE ---
def prepare_input(feature_list):
    """ 
    Ensures input matches the model's 'fit' schema perfectly:
    - Same columns
    - Same order
    - No missing features
    """
    # 1. Start with a dictionary of 0s based on the model's trained feature names
    input_dict = {col: 0 for col in feature_list}

    # 2. Map user UI values to the dictionary
    mapping = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'PercentSalaryHike': hike,
        'YearsAtCompany': years_at_co,
        'TotalWorkingYears': total_work_years,
        'JobSatisfaction': job_sat,
        'EnvironmentSatisfaction': env_sat,
        'JobInvolvement': job_inv,
        'DistanceFromHome': dist_home
    }
    
    for key, val in mapping.items():
        if key in input_dict:
            input_dict[key] = val

    # 3. Handle One-Hot Encoding (Logic must match training)
    if overtime == "Yes" and 'OverTime_1' in input_dict:
        input_dict['OverTime_1'] = 1
    
    if marital == "Married" and 'MaritalStatus_Married' in input_dict:
        input_dict['MaritalStatus_Married'] = 1
    elif marital == "Single" and 'MaritalStatus_Single' in input_dict:
        input_dict['MaritalStatus_Single'] = 1
        
    dept_col = f"Department_{dept}"
    if dept_col in input_dict: 
        input_dict[dept_col] = 1

    role_col = f"JobRole_{job_role}"
    if role_col in input_dict: 
        input_dict[role_col] = 1

    # 4. Convert to DataFrame
    df_temp = pd.DataFrame([input_dict])
    
    # 5. CRITICAL STEP: Reindex forces the DataFrame to match the 
    # feature_list order and names exactly as seen during model fit.
    return df_temp.reindex(columns=feature_list).fillna(0)

# --- RESULTS TABS ---
st.divider()
tab1, tab2 = st.tabs(["🚪 Attrition Analysis", "📈 Performance Prediction"])

with tab1:
    st.subheader("Predict Probability of Turnover")
    if st.button("Predict Attrition"):
        # Process data for the Attrition model
        input_df = prepare_input(attr_features)
        
        # Inference
        prob = attr_model.predict_proba(input_df)[0][1]
        
        if prob > 0.5:
            st.error(f"### Attrition Risk: {prob:.2%}")
        else:
            st.success(f"### Attrition Risk: {prob:.2%}")
        st.progress(prob)

        # Show built-in bar chart of importance
        if attr_importance_df is not None:
            st.markdown("---")
            st.write("Model Decision Factors (Top 10):")
            st.bar_chart(attr_importance_df.head(10))

with tab2:
    st.subheader("Predict Future Performance Rating")
    if st.button("Calculate Rating"):
        # Process data for the Performance model
        input_df = prepare_input(perf_features)
        
        # Scaling is required for Linear Regression
        scaled_input = scaler.transform(input_df)
        rating = perf_model.predict(scaled_input)[0]
        
        st.metric("Expected Performance Rating", f"{rating:.2f}")

# --- DIAGNOSTIC PREVIEW ---
with st.expander("🔍 View Processed Model Input"):
    st.write("This shows exactly what is being sent to the model (Column Order Check):")
    st.dataframe(prepare_input(attr_features))
