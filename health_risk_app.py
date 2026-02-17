import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("enhanced_health_data.csv")
    # Convert 'Health' to binary risk (0 = Low, 1 = High)
    data['Risk'] = data['Health'].apply(lambda x: 1 if x == 'Bad' else 0)
    return data

data = load_data()

# Train a simple model
def train_model(data):
    # Select features and target
    features = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'BMI', 'Smoker', 'Diabetes']
    X = data[features]
    y = data['Risk']
    
    # Convert boolean to int
    X['Smoker'] = X['Smoker'].astype(int)
    X['Diabetes'] = X['Diabetes'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model(data)

# Streamlit app
st.title("Health Risk Assessment Tool")
st.write("This app predicts whether a patient is at Low Risk or High Risk based on health metrics.")

st.sidebar.header("Patient Information")

# Input fields
age = st.sidebar.slider("Age", 18, 100, 40)
systolic_bp = st.sidebar.slider("Systolic Blood Pressure", 90, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic Blood Pressure", 60, 120, 80)
cholesterol = st.sidebar.slider("Cholesterol Level", 100, 300, 200)
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
smoker = st.sidebar.radio("Smoker", ["No", "Yes"])
diabetes = st.sidebar.radio("Diabetes", ["No", "Yes"])

# Convert inputs to model format
smoker = 1 if smoker == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, cholesterol, bmi, smoker, diabetes]],
                         columns=['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'BMI', 'Smoker', 'Diabetes'])

# Make prediction
if st.sidebar.button("Assess Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Risk Assessment Result")
    if prediction == 0:
        st.success("Low Risk")
        st.write(f"Probability of being High Risk: {probability*100:.1f}%")
    else:
        st.error("High Risk")
        st.write(f"Probability of being High Risk: {probability*100:.1f}%")
    
    # Show some interpretation
    st.subheader("Risk Factors")
    if age > 60:
        st.write("- Advanced age increases risk")
    if systolic_bp > 140 or diastolic_bp > 90:
        st.write("- Elevated blood pressure increases risk")
    if cholesterol > 240:
        st.write("- High cholesterol increases risk")
    if bmi > 30:
        st.write("- Obesity (BMI > 30) increases risk")
    if smoker:
        st.write("- Smoking increases risk")
    if diabetes:
        st.write("- Diabetes increases risk")

# Show model info
st.sidebar.subheader("Model Information")
st.sidebar.write(f"Model Accuracy: {accuracy*100:.1f}%")
st.sidebar.write("Based on analysis of similar patient data")

# Show sample data
if st.checkbox("Show sample data"):
    st.subheader("Sample Health Data")
    st.write(data.head(10))