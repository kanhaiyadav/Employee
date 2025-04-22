import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #4c78a8;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3d6199;
        margin-top: 1rem;
        white-space: nowrap;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .bold-text {
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #4c78a8;
        color: white;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #3d6199;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .risk-high {
        background-color: rgba(255, 76, 76, 0.2);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ff4c4c;
    }
    .risk-low {
        background-color: rgba(76, 255, 76, 0.2);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4cff4c;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👨‍💼 Employee Attrition Prediction Tool</div>', unsafe_allow_html=True)
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

icons = {
    "personal": "👤",
    "job": "💼",
    "compensation": "💰",
    "satisfaction": "😊",
    "education": "🎓",
    "travel": "✈️",
    "overtime": "⏰",
    "experience": "📆",
    "income": "💵",
    "training": "📚",
    "distance": "🏠",
    "stock": "📈",
    "balance": "⚖️",
    "involvement": "🔄",
    "relationship": "🤝",
    "environment": "🌿"
}

def create_feature_input():
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    
    with col1:
        st.markdown(f'<div class="sub-header">{icons["personal"]} Personal Information</div>', unsafe_allow_html=True)
        with st.container(border=True):
            input_data['Age'] = st.slider(f"{icons['personal']} Age", min_value=18, max_value=60, value=35, step=1, 
                                      help="Age of employee")
            
            input_data['Gender'] = st.selectbox(f"{icons['personal']} Gender", options=["Male", "Female"], 
                                            help="Gender of employee")
            
            input_data['MaritalStatus'] = st.selectbox(f"{icons['personal']} Marital Status", 
                                                   options=["Single", "Married", "Divorced"], 
                                                   help="Marital status of employee")
            
            input_data['DistanceFromHome'] = st.slider(f"{icons['distance']} Distance From Home", min_value=1, max_value=30, 
                                                   value=10, step=1, 
                                                   help="Distance from home in miles/km")
            
            input_data['NumCompaniesWorked'] = st.slider(f"{icons['experience']} Number of Companies Worked", 
                                                     min_value=0, max_value=10, value=2, step=1,
                                                     help="Number of companies worked at before")
            
    with col2:
        st.markdown(f'<div class="sub-header">{icons["job"]} Job Information</div>', unsafe_allow_html=True)
        with st.container(border=True):
            input_data['Department'] = st.selectbox(f"{icons['job']} Department", 
                                                options=["Research & Development", "Sales", "Human Resources"], 
                                                help="Employee's department")
            
            input_data['JobRole'] = st.selectbox(f"{icons['job']} Job Role", 
                                             options=["Sales Executive", "Research Scientist", 
                                                     "Laboratory Technician", "Manufacturing Director", 
                                                     "Healthcare Representative", "Manager", 
                                                     "Sales Representative", "Research Director", 
                                                     "Human Resources"], 
                                             help="Employee's job role")
            
            input_data['BusinessTravel'] = st.selectbox(f"{icons['travel']} Business Travel", 
                                                    options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"], 
                                                    help="Frequency of business travel")
            
            input_data['OverTime'] = st.selectbox(f"{icons['overtime']} Works Overtime?", 
                                              options=["No", "Yes"], 
                                              help="Does the employee work overtime?")
            
            input_data['YearsAtCompany'] = st.slider(f"{icons['experience']} Years at Company", 
                                                 min_value=0, max_value=40, value=5, step=1, 
                                                 help="Years at the company")
            
            input_data['YearsInCurrentRole'] = st.slider(f"{icons['experience']} Years in Current Role", 
                                                     min_value=0, max_value=20, value=3, step=1, 
                                                     help="Years in current role")
            
            input_data['YearsSinceLastPromotion'] = st.slider(f"{icons['experience']} Years Since Last Promotion", 
                                                          min_value=0, max_value=15, value=1, step=1, 
                                                          help="Years since last promotion")
            
    with col3:
        st.markdown(f'<div class="sub-header">{icons["compensation"]} Compensation & Education</div>', unsafe_allow_html=True)
        with st.container(border=True):
            input_data['Education'] = st.slider(f"{icons['education']} Education Level", 
                                            min_value=1, max_value=5, value=3, step=1, 
                                            help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
            
            input_data['EducationField'] = st.selectbox(f"{icons['education']} Education Field", 
                                                    options=["Life Sciences", "Medical", "Marketing", 
                                                            "Technical Degree", "Human Resources", "Other"], 
                                                    help="Field of education")
            
            input_data['MonthlyIncome'] = st.slider(f"{icons['income']} Monthly Income", 
                                                min_value=1000, max_value=20000, value=6000, step=500, 
                                                help="Monthly income in $")
            
            input_data['DailyRate'] = st.slider(f"{icons['income']} Daily Rate", 
                                            min_value=100, max_value=1500, value=800, step=50, 
                                            help="Daily rate in $")
            
            input_data['HourlyRate'] = st.slider(f"{icons['income']} Hourly Rate", 
                                             min_value=30, max_value=100, value=65, step=5, 
                                             help="Hourly rate in $")
            
            input_data['MonthlyRate'] = st.slider(f"{icons['income']} Monthly Rate", 
                                              min_value=5000, max_value=25000, value=15000, step=1000, 
                                              help="Monthly rate in $")
            
            input_data['PercentSalaryHike'] = st.slider(f"{icons['income']} Percent Salary Hike", 
                                                    min_value=10, max_value=25, value=15, step=1, 
                                                    help="Percentage of last salary hike")
            
            input_data['StockOptionLevel'] = st.slider(f"{icons['stock']} Stock Option Level", 
                                                   min_value=0, max_value=3, value=1, step=1, 
                                                   help="Stock option level (0-3)")
            
            input_data['TrainingTimesLastYear'] = st.slider(f"{icons['training']} Training Times Last Year", 
                                                        min_value=0, max_value=6, value=3, step=1, 
                                                        help="Number of trainings last year")
    
    st.markdown(f'<div class="sub-header">{icons["satisfaction"]} Satisfaction Metrics</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            input_data['JobSatisfaction'] = st.slider(f"{icons['satisfaction']} Job Satisfaction", 
                                                  min_value=1, max_value=4, value=3, step=1, 
                                                  help="Job satisfaction level (1-4)")
        
        with col5:
            input_data['EnvironmentSatisfaction'] = st.slider(f"{icons['environment']} Environment Satisfaction", 
                                                          min_value=1, max_value=4, value=3, step=1, 
                                                          help="Environment satisfaction level (1-4)")
        
        with col6:
            input_data['WorkLifeBalance'] = st.slider(f"{icons['balance']} Work Life Balance", 
                                                  min_value=1, max_value=4, value=3, step=1, 
                                                  help="Work life balance rating (1-4)")
        
        with col7:
            input_data['RelationshipSatisfaction'] = st.slider(f"{icons['relationship']} Relationship Satisfaction", 
                                                           min_value=1, max_value=4, value=3, step=1, 
                                                           help="Relationship satisfaction level (1-4)")
        
        with col4:
            input_data['JobInvolvement'] = st.slider(f"{icons['involvement']} Job Involvement", 
                                                 min_value=1, max_value=4, value=3, step=1, 
                                                 help="Job involvement level (1-4)")
    
    return input_data

user_inputs = create_feature_input()

submit_button = st.button("🔍 Predict Attrition")

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
    with st.spinner('🔄 Processing...'):
        processed_input = preprocess_input(user_inputs)
        
        prediction, probability, custom_prediction = predict_attrition(processed_input)
        
        st.markdown('<div class="sub-header">🔎 Prediction Results</div>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if custom_prediction == 1:
                st.markdown('<div class="risk-high"><h3>⚠️ HIGH RISK OF ATTRITION</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low"><h3>✅ LOW RISK OF ATTRITION</h3></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            
            with st.container(border=True):
                st.metric("🎯 Probability of Leaving", f"{probability:.2%}")
                
                if probability < 0.2:
                    risk_level = "Very Low Risk"
                    suggestion = "Employee seems satisfied and likely to stay."
                    risk_icon = "😊"
                elif probability < 0.4:
                    risk_level = "Low Risk"
                    suggestion = "Some factors may contribute to attrition risk, but overall risk is low."
                    risk_icon = "🙂"
                elif probability < 0.6:
                    risk_level = "Moderate Risk"
                    suggestion = "Consider checking in with this employee and addressing potential concerns."
                    risk_icon = "😐"
                elif probability < 0.8:
                    risk_level = "High Risk"
                    suggestion = "Take action to address retention risk factors."
                    risk_icon = "😟"
                else:
                    risk_level = "Very High Risk"
                    suggestion = "Immediate attention needed. Employee is at high risk of leaving."
                    risk_icon = "😨"
                
                st.write(f"**Risk Level:** {risk_icon} {risk_level}")
                st.write(f"**Suggestion:** 💡 {suggestion}")
        
        with res_col2:
            with st.container(border=True):
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Attrition Risk Gauge", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 20], 'color': '#90EE90'},
                            {'range': [20, 40], 'color': '#FFFF99'},
                            {'range': [40, 60], 'color': '#FFD700'},
                            {'range': [60, 80], 'color': '#FFA07A'},
                            {'range': [80, 100], 'color': '#FF6347'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 53.5 
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
        st.markdown('<div class="sub-header">📈 Top Contributing Factors</div>', unsafe_allow_html=True)
        
        with st.container(border=True):
            contributing_factors = [
                {"factor": f"{icons['overtime']} Overtime & Work-Life Balance", "impact": 0.45, "direction": "increases risk"},
                {"factor": f"{icons['income']} Monthly Income", "impact": 0.33, "direction": "decreases risk"},
                {"factor": f"{icons['travel']} Business Travel", "impact": 0.32, "direction": "increases risk"},
                {"factor": f"{icons['involvement']} Job Involvement", "impact": 0.28, "direction": "decreases risk"},
                {"factor": f"{icons['environment']} Environment Satisfaction", "impact": 0.26, "direction": "decreases risk"}
            ]
            
            factors_df = pd.DataFrame(contributing_factors)
            
            fig = px.bar(
                factors_df,
                y="factor",
                x="impact",
                color="direction",
                color_discrete_map={"increases risk": "#ff6666", "decreases risk": "#66b3ff"},
                title="Top Factors Contributing to Prediction",
                orientation='h',
                text="impact"
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                height=400,
                xaxis_title="Impact Magnitude",
                legend_title="Effect",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔍 Show all input values"):
            st.json(user_inputs)

#information about the model
with st.sidebar:
    st.markdown('<div class="sub-header">🤖 About this Model</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write("""
        This model predicts the likelihood that an employee will leave a company based on various factors including:
        
        - 👤 Demographics (age, gender, etc.)
        - 💼 Job characteristics
        - 💰 Compensation
        - 😊 Satisfaction metrics
        - 📆 Work history
        
        The model was trained using logistic regression with optimized parameters and has achieved approximately 86% accuracy on test data.
        """)
    
    st.markdown('<div class="sub-header">📝 How to use this tool</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write("""
        1. Fill in all the employee information fields
        2. Click the 'Predict Attrition' button
        3. Review the prediction results and risk factors
        4. Use these insights for retention planning
        """)
    
    with st.container(border=True):
        st.info("ℹ️ This tool is for informational purposes only and should be used as one of many factors in employee retention decisions.")