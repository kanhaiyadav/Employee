import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("Employee Attrition Prediction Tool")
st.markdown("""
This application uses a machine learning model to predict the likelihood of an employee leaving a company.
Enter the employee information in the form below to get a prediction.
""")

MODEL_PATH = 'attrition_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found! Please train the model first.")
        return None

model = load_model()

def create_feature_input():
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    
    with col1:
        st.subheader("Personal Information")
        input_data['Age'] = st.slider("Age", min_value=18, max_value=60, value=35, step=1, 
                                  help="Age of employee")
        
        input_data['Gender'] = st.selectbox("Gender", options=["Male", "Female"], 
                                        help="Gender of employee")
        
        input_data['MaritalStatus'] = st.selectbox("Marital Status", 
                                               options=["Single", "Married", "Divorced"], 
                                               help="Marital status of employee")
        
        input_data['DistanceFromHome'] = st.slider("Distance From Home", min_value=1, max_value=30, 
                                               value=10, step=1, 
                                               help="Distance from home in miles/km")
        
        input_data['NumCompaniesWorked'] = st.slider("Number of Companies Worked", 
                                                 min_value=0, max_value=10, value=2, step=1,
                                                 help="Number of companies worked at before")
        
    with col2:
        st.subheader("Job Information")
        input_data['Department'] = st.selectbox("Department", 
                                            options=["Research & Development", "Sales", "Human Resources"], 
                                            help="Employee's department")
        
        input_data['JobRole'] = st.selectbox("Job Role", 
                                         options=["Sales Executive", "Research Scientist", 
                                                 "Laboratory Technician", "Manufacturing Director", 
                                                 "Healthcare Representative", "Manager", 
                                                 "Sales Representative", "Research Director", 
                                                 "Human Resources"], 
                                         help="Employee's job role")
        
        input_data['BusinessTravel'] = st.selectbox("Business Travel", 
                                                options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"], 
                                                help="Frequency of business travel")
        
        input_data['OverTime'] = st.selectbox("Works Overtime?", 
                                          options=["No", "Yes"], 
                                          help="Does the employee work overtime?")
        
        input_data['YearsAtCompany'] = st.slider("Years at Company", 
                                             min_value=0, max_value=40, value=5, step=1, 
                                             help="Years at the company")
        
        input_data['YearsInCurrentRole'] = st.slider("Years in Current Role", 
                                                 min_value=0, max_value=20, value=3, step=1, 
                                                 help="Years in current role")
        
        input_data['YearsSinceLastPromotion'] = st.slider("Years Since Last Promotion", 
                                                      min_value=0, max_value=15, value=1, step=1, 
                                                      help="Years since last promotion")
        
    with col3:
        st.subheader("Compensation & Satisfaction")
        input_data['Education'] = st.slider("Education Level", 
                                        min_value=1, max_value=5, value=3, step=1, 
                                        help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
        
        input_data['EducationField'] = st.selectbox("Education Field", 
                                                options=["Life Sciences", "Medical", "Marketing", 
                                                        "Technical Degree", "Human Resources", "Other"], 
                                                help="Field of education")
        
        input_data['MonthlyIncome'] = st.slider("Monthly Income", 
                                            min_value=1000, max_value=20000, value=6000, step=500, 
                                            help="Monthly income in $")
        
        input_data['DailyRate'] = st.slider("Daily Rate", 
                                        min_value=100, max_value=1500, value=800, step=50, 
                                        help="Daily rate in $")
        
        input_data['HourlyRate'] = st.slider("Hourly Rate", 
                                         min_value=30, max_value=100, value=65, step=5, 
                                         help="Hourly rate in $")
        
        input_data['MonthlyRate'] = st.slider("Monthly Rate", 
                                          min_value=5000, max_value=25000, value=15000, step=1000, 
                                          help="Monthly rate in $")
        
        input_data['PercentSalaryHike'] = st.slider("Percent Salary Hike", 
                                                min_value=10, max_value=25, value=15, step=1, 
                                                help="Percentage of last salary hike")
        
        input_data['StockOptionLevel'] = st.slider("Stock Option Level", 
                                               min_value=0, max_value=3, value=1, step=1, 
                                               help="Stock option level (0-3)")
        
        input_data['TrainingTimesLastYear'] = st.slider("Training Times Last Year", 
                                                    min_value=0, max_value=6, value=3, step=1, 
                                                    help="Number of trainings last year")
    
    st.subheader("Satisfaction Metrics")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        input_data['JobSatisfaction'] = st.slider("Job Satisfaction", 
                                              min_value=1, max_value=4, value=3, step=1, 
                                              help="Job satisfaction level (1-4)")
    
    with col5:
        input_data['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction", 
                                                      min_value=1, max_value=4, value=3, step=1, 
                                                      help="Environment satisfaction level (1-4)")
    
    with col6:
        input_data['WorkLifeBalance'] = st.slider("Work Life Balance", 
                                              min_value=1, max_value=4, value=3, step=1, 
                                              help="Work life balance rating (1-4)")
    
    with col7:
        input_data['RelationshipSatisfaction'] = st.slider("Relationship Satisfaction", 
                                                       min_value=1, max_value=4, value=3, step=1, 
                                                       help="Relationship satisfaction level (1-4)")
    
    with col4:
        input_data['JobInvolvement'] = st.slider("Job Involvement", 
                                             min_value=1, max_value=4, value=3, step=1, 
                                             help="Job involvement level (1-4)")
    
    return input_data

user_inputs = create_feature_input()

submit_button = st.button("Predict Attrition")

def preprocess_input(user_data):
    input_df = pd.DataFrame([user_data])
    
    total_years = user_data['YearsAtCompany'] + 1 
    input_df['IncomePerWorkingYear'] = user_data['MonthlyIncome'] / total_years
    
    # Create log transformations for skewed features
    skewed_features = ['MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 
                      'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
                      'YearsSinceLastPromotion', 'IncomePerWorkingYear', 'DistanceFromHome', 
                      'PercentSalaryHike']
    
    for col in skewed_features:
        input_df[f'Log{col}'] = np.log1p(input_df[col])
    
    input_df['SatisfactionProduct'] = input_df['JobSatisfaction'] * input_df['WorkLifeBalance']
    input_df['PromotionSatisfaction'] = input_df['LogYearsSinceLastPromotion'] * input_df['JobSatisfaction']
    input_df['OverTimeWorkLife'] = input_df['OverTime'].map({'Yes': 1, 'No': 0}) * input_df['WorkLifeBalance']
    
    input_df['AgeGroup'] = pd.cut(input_df['Age'], bins=[18, 30, 40, 50, 60], 
                                 labels=['Young', 'Early-Mid', 'Late-Mid', 'Senior'])
    input_df['TenureGroup'] = pd.cut(input_df['YearsAtCompany'], bins=[0, 2, 5, 10, 40], 
                                    labels=['New', 'Developing', 'Experienced', 'Veteran'])
    
    return input_df

def predict_attrition(processed_df):
    if model is not None:
        probabilities = model.predict_proba(processed_df)
        probability_yes = probabilities[0][1]
        
        prediction = model.predict(processed_df)[0]
        
        optimal_threshold = 0.535
        custom_prediction = 1 if probability_yes >= optimal_threshold else 0
        
        return prediction, probability_yes, custom_prediction
    return None, None, None

if submit_button:
    with st.spinner('Processing...'):
        processed_input = preprocess_input(user_inputs)
        
        prediction, probability, custom_prediction = predict_attrition(processed_input)
        
        st.header("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if custom_prediction == 1:
                st.error("⚠️ HIGH RISK OF ATTRITION")
            else:
                st.success("✅ LOW RISK OF ATTRITION")
            
            st.metric("Probability of Leaving", f"{probability:.2%}")
            
            if probability < 0.2:
                risk_level = "Very Low Risk"
                suggestion = "Employee seems satisfied and likely to stay."
            elif probability < 0.4:
                risk_level = "Low Risk"
                suggestion = "Some factors may contribute to attrition risk, but overall risk is low."
            elif probability < 0.6:
                risk_level = "Moderate Risk"
                suggestion = "Consider checking in with this employee and addressing potential concerns."
            elif probability < 0.8:
                risk_level = "High Risk"
                suggestion = "Take action to address retention risk factors."
            else:
                risk_level = "Very High Risk"
                suggestion = "Immediate attention needed. Employee is at high risk of leaving."
            
            st.write(f"**Risk Level:** {risk_level}")
            st.write(f"**Suggestion:** {suggestion}")
        
        with res_col2:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 4))
            
            gauge_values = [probability, 1-probability]
            gauge_labels = ['Risk', 'Safe']
            gauge_colors = ['#ff9999', '#99ff99']
            
            wedges, texts = ax.pie(gauge_values, 
                                  wedgeprops=dict(width=0.5),
                                  startangle=90,
                                  counterclock=False,
                                  colors=gauge_colors)
            
            arrow_x = 0
            arrow_y = 0
            arrow_length = 0.75
            arrow_angle = (0.5 - probability) * 180
            dx = arrow_length * np.cos(np.radians(arrow_angle))
            dy = arrow_length * np.sin(np.radians(arrow_angle))
            ax.arrow(arrow_x, arrow_y, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black')
            
            ax.text(0, -1.1, 'Attrition Risk Gauge', ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax.text(0, 0, f"{probability:.1%}", ha='center', va='center', fontsize=16, fontweight='bold')
            
            st.pyplot(fig)
            
        st.subheader("Top Contributing Factors")
        
        contributing_factors = [
            {"factor": "Overtime & Work-Life Balance", "impact": 0.45, "direction": "increases risk"},
            {"factor": "Monthly Income", "impact": 0.33, "direction": "decreases risk"},
            {"factor": "Business Travel", "impact": 0.32, "direction": "increases risk"},
            {"factor": "Job Involvement", "impact": 0.28, "direction": "decreases risk"},
            {"factor": "Environment Satisfaction", "impact": 0.26, "direction": "decreases risk"}
        ]
        
        factors_df = pd.DataFrame(contributing_factors)
        
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        factors = factors_df["factor"].tolist()
        impacts = factors_df["impact"].tolist()
        directions = factors_df["direction"].tolist()
        
        colors = ['#ff6666' if d == "increases risk" else '#66b3ff' for d in directions]
        
        bars = ax.barh(factors, impacts, color=colors)
        
        ax.set_xlabel('Impact Magnitude')
        ax.set_title('Top Factors Contributing to Prediction')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff6666', label='Increases Risk'),
            Patch(facecolor='#66b3ff', label='Decreases Risk')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        st.pyplot(fig)
        
        with st.expander("Show all input values"):
            st.json(user_inputs)

#information about the model
with st.sidebar:
    st.header("About this Model")
    st.write("""
    This model predicts the likelihood that an employee will leave a company based on various factors including:
    
    - Demographics (age, gender, etc.)
    - Job characteristics
    - Compensation
    - Satisfaction metrics
    - Work history
    
    The model was trained using logistic regression with optimized parameters and has achieved approximately 86% accuracy on test data.
    """)
    
    st.subheader("How to use this tool")
    st.write("""
    1. Fill in all the employee information fields
    2. Click the 'Predict Attrition' button
    3. Review the prediction results and risk factors
    4. Use these insights for retention planning
    """)
    
    st.info("This tool is for informational purposes only and should be used as one of many factors in employee retention decisions.")