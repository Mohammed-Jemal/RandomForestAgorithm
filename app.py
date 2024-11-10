import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Health Prediction App", layout="wide")

# Load the saved model and artifacts
@st.cache_resource
def load_model():
    artifacts = joblib.load('health_care_ML.joblib')
    return artifacts

try:
    artifacts = load_model()
    model = artifacts['model']
    feature_names = artifacts['feature_names']
    feature_importance = artifacts['feature_importance']
except:
    st.error("Please ensure 'health_care_ML.joblib' is in the same directory as this script")
    st.stop()

# Title and description
st.title('🏥 Health Condition Predictor')
st.markdown("""
This app predicts health conditions based on various health metrics and lifestyle factors.
Please fill in the information below to get a prediction.
""")

# Create input form
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Numerical inputs
    age = st.sidebar.slider('Age', 18, 100, 30)
    bmi = st.sidebar.slider('BMI', 15.0, 50.0, 25.0)
    blood_pressure = st.sidebar.slider('Blood Pressure', 80, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol', 100, 300, 200)
    glucose = st.sidebar.slider('Glucose Level', 70, 200, 100)
    heart_rate = st.sidebar.slider('Heart Rate', 60, 120, 80)
    sleep_hours = st.sidebar.slider('Sleep Hours', 4, 12, 7)
    exercise_hours = st.sidebar.slider('Exercise Hours', 0, 10, 2)
    water_intake = st.sidebar.slider('Water Intake (Liters)', 0.0, 5.0, 2.0)
    stress_level = st.sidebar.slider('Stress Level', 0, 10, 5)
    
    # Categorical inputs
    smoking = st.sidebar.selectbox('Smoking', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    alcohol = st.sidebar.selectbox('Alcohol Consumption', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    diet = st.sidebar.selectbox('Diet Quality', [0, 1, 2, 3], format_func=lambda x: ['Poor', 'Fair', 'Good', 'Excellent'][x])
    mental_health = st.sidebar.selectbox('Mental Health', [0, 1, 2, 3], format_func=lambda x: ['Poor', 'Fair', 'Good', 'Excellent'][x])
    physical_activity = st.sidebar.selectbox('Physical Activity Level', [0, 1, 2, 3], format_func=lambda x: ['Sedentary', 'Low', 'Moderate', 'High'][x])
    medical_history = st.sidebar.selectbox('Medical History', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    allergies = st.sidebar.selectbox('Allergies', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    # Diet type and blood group
    diet_vegan = st.sidebar.checkbox('Vegan Diet')
    diet_vegetarian = st.sidebar.checkbox('Vegetarian Diet')
    blood_group_ab = st.sidebar.checkbox('Blood Group AB')
    blood_group_b = st.sidebar.checkbox('Blood Group B')
    blood_group_o = st.sidebar.checkbox('Blood Group O')
    
    data = {
        'Age': age,
        'BMI': bmi,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Glucose_Level': glucose,
        'Heart_Rate': heart_rate,
        'Sleep_Hours': sleep_hours,
        'Exercise_Hours': exercise_hours,
        'Water_Intake': water_intake,
        'Stress_Level': stress_level,
        'Smoking': smoking,
        'Alcohol': alcohol,
        'Diet': diet,
        'MentalHealth': mental_health,
        'PhysicalActivity': physical_activity,
        'MedicalHistory': medical_history,
        'Allergies': allergies,
        'Diet_Type__Vegan': diet_vegan,
        'Diet_Type__Vegetarian': diet_vegetarian,
        'Blood_Group_AB': blood_group_ab,
        'Blood_Group_B': blood_group_b,
        'Blood_Group_O': blood_group_o
    }
    return pd.DataFrame(data, index=[0])

# Get user input
user_data = user_input_features()

# Main panel
st.header('Specified Input Parameters')
st.write(user_data)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)
    
    # Display prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Prediction')
        prediction_text = 'Healthy' if prediction[0] == 0 else 'Health Risk'
        prediction_color = 'green' if prediction[0] == 0 else 'red'
        st.markdown(f'<h1 style="color: {prediction_color};">{prediction_text}</h1>', 
                   unsafe_allow_html=True)

    with col2:
        st.subheader('Prediction Probability')
        prob_fig = go.Figure(go.Bar(
            x=['Healthy', 'Health Risk'],
            y=[prediction_proba[0][0], prediction_proba[0][1]],
            marker_color=['green', 'red']
        ))
        prob_fig.update_layout(
            yaxis_title='Probability',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(prob_fig)

# Feature Importance Plot
st.header('Feature Importance')
fig = px.bar(feature_importance.head(10), 
             x='importance', 
             y='feature',
             orientation='h',
             title='Top 10 Most Important Features')
st.plotly_chart(fig)

# Add explanatory notes
st.markdown("""
### Notes:
- The prediction is based on the Random Forest model trained on healthcare data
- Feature importance shows which factors most strongly influence the prediction
- The probability chart shows the model's confidence in its prediction
""")

# Add footer
st.markdown("""
---
Created by [Mohammed Jemal] | [GitHub](Mohammed-Jemal/RandomForestAgorithm)
""")
