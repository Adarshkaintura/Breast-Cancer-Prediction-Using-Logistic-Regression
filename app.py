import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('archive/data.csv')

# Data preprocessing
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
df["diagnosis"] = [1 if value == "M" else 0 for value in df.diagnosis]
df["diagnosis"] = df['diagnosis'].astype("category", copy=False)

# Features and target variable
y = df["diagnosis"]
X = df.drop(["diagnosis"], axis=1)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Streamlit begins from here
st.title("Breast Cancer Prediction")

# Create input fields for the user to enter all the required attributes
attributes = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# taking all the 30 parameters
user_input = []
for attr in attributes:
    value = st.number_input(f'{attr.replace("_", " ").title()}', min_value=0.0, value=1.0)
    user_input.append(value)

# Standardization for values to make them in same unit
user_input_scaled = scaler.transform([user_input])

# now prediction
prediction = lr.predict(user_input_scaled)

# result display
if prediction == 1:
    st.write("Prediction: Cancerous")
else:
    st.write("Prediction: Non-Cancerous")
