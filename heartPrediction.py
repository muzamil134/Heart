import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Model accuracy
log_acc = accuracy_score(y_test, log_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# Streamlit UI
st.title("❤ Heart Disease Prediction App")
st.write("Enter your health details below to check if you might have heart disease.")

# Take user input
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", [1, 0])
restecg = st.number_input("Resting ECG (0-2)", 0, 2)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
slope = st.number_input("Slope (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.number_input("Thal (0-3)", 0, 3)

# Combine inputs into dataframe
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=X.columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Using Logistic Regression"):
    prediction = log_model.predict(input_scaled)
    result = "💔 Heart Disease Detected" if prediction[0] == 1 else "❤ No Heart Disease"
    st.success(result)
    st.write("Model Accuracy:", round(log_acc*100, 2), "%")

if st.button("Predict Using Random Forest"):
    prediction = rf_model.predict(input_scaled)
    result = "💔 Heart Disease Detected" if prediction[0] == 1 else "❤ No Heart Disease"
    st.success(result)
    st.write("Model Accuracy:", round(rf_acc*100, 2), "%")