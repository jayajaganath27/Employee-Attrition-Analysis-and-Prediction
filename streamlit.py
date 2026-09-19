import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💼",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    color: #4B0082;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 18px;
}

.prediction-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD SAVED MODEL FILES
# ============================================================

model = joblib.load(r"c:\Users\dell\Desktop\Jaya - Data Science\Project\employee attribution\Attrition Prediction\gradient_boosting_best.pkl")

encoder = joblib.load(r"C:\Users\dell\Desktop\Jaya - Data Science\Project\employee attribution\Attrition Prediction\employee_encoder.pkl")

feature_names = joblib.load(r"C:\Users\dell\Desktop\Jaya - Data Science\Project\employee attribution\Attrition Prediction\gb_feature_names.pkl")


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">👩‍💼 Employee Attrition Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether an employee is likely to leave the organization'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# EMPLOYEE INFORMATION
# ============================================================

st.header("📋 Employee Information")

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=30000,
        value=5000
    )

    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1, 2, 3, 4]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "Yes",
            "No"
        ]
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=15
    )

    performance_rating = st.selectbox(
        "Performance Rating",
        [1, 2, 3, 4]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=3
    )

    years_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=2
    )

    years_since_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    years_with_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=2
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.subheader("⚙️ Feature Engineering")

# ------------------------------------------------------------
# Daily Rate Input
# ------------------------------------------------------------

daily_rate = st.number_input(
    "Daily Rate",
    min_value=100,
    max_value=1500,
    value=800
)


# ------------------------------------------------------------
# Daily Rate Group
# ------------------------------------------------------------

if daily_rate < 500:

    daily_rate_group = "Low"

elif daily_rate < 1000:

    daily_rate_group = "Medium"

else:

    daily_rate_group = "High"


# ------------------------------------------------------------
# Distance Group
# ------------------------------------------------------------

if distance_from_home <= 5:

    distance_group = "Very Near"

elif distance_from_home <= 10:

    distance_group = "Near"

elif distance_from_home <= 20:

    distance_group = "Far"

else:

    distance_group = "Very Far"


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

if st.button(
    "🔮 Predict Attrition",
    use_container_width=True
):

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    input_data = pd.DataFrame({

        "Age": [age],

        "DistanceFromHome": [
            distance_from_home
        ],

        "Education": [
            education
        ],

        "EnvironmentSatisfaction": [
            environment_satisfaction
        ],

        "JobInvolvement": [
            job_involvement
        ],

        "JobLevel": [
            job_level
        ],

        "JobSatisfaction": [
            job_satisfaction
        ],

        "MonthlyIncome": [
            monthly_income
        ],

        "NumCompaniesWorked": [
            num_companies_worked
        ],

        "OverTime": [
            1 if overtime == "Yes" else 0
        ],

        "PercentSalaryHike": [
            percent_salary_hike
        ],

        "PerformanceRating": [
            performance_rating
        ],

        "RelationshipSatisfaction": [
            relationship_satisfaction
        ],

        "TotalWorkingYears": [
            total_working_years
        ],

        "YearsAtCompany": [
            years_at_company
        ],

        "YearsInCurrentRole": [
            years_current_role
        ],

        "YearsSinceLastPromotion": [
            years_since_promotion
        ],

        "YearsWithCurrManager": [
            years_with_manager
        ]
    })


    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_data = pd.DataFrame({

        "BusinessTravel": [
            business_travel
        ],

        "Department": [
            department
        ],

        "EducationField": [
            education_field
        ],

        "Gender": [
            gender
        ],

        "JobRole": [
            job_role
        ],

        "MaritalStatus": [
            marital_status
        ],

        "DailyRate_Group": [
            daily_rate_group
        ],

        "Distance_Group": [
            distance_group
        ]
    })


    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    encoded_data = encoder.transform(
        categorical_data
    )


    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            categorical_data.columns
        )
    )


    # ========================================================
    # COMBINE NUMERIC + ENCODED FEATURES
    # ========================================================

    final_input = pd.concat(
        [
            input_data.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )


    # ========================================================
    # REORDER FEATURES
    # ========================================================

    final_input = final_input.reindex(
        columns=feature_names,
        fill_value=0
    )


    # ========================================================
    # CHECK FEATURE COUNT
    # ========================================================

    if final_input.shape[1] != model.n_features_in_:

        st.error(
            f"Feature mismatch! "
            f"Model expects {model.n_features_in_} features, "
            f"but received {final_input.shape[1]} features."
        )

        st.stop()


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(final_input)[0]

    probability = model.predict_proba(final_input)[0][1]


    # ========================================================
    # DISPLAY PREDICTION RESULT
    # ========================================================

    st.subheader(
        "📊 Prediction Result"
    )


    # --------------------------------------------------------
    # HIGH ATTRITION RISK
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ High Attrition Risk: "
            "Employee is likely to leave."
        )

        st.write(
            f"Attrition Probability: "
            f"**{probability:.2%}**"
        )


    # --------------------------------------------------------
    # LOW ATTRITION RISK
    # --------------------------------------------------------

    else:

        st.success(
            "✅ Low Attrition Risk: "
            "Employee is likely to stay."
        )

        st.write(
            f"Attrition Probability: "
            f"**{probability:.2%}**"
        )


 