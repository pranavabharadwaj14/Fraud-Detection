from PIL import Image
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load and preprocess data
data = pd.read_excel("Book1.xlsx")
data = data.drop(columns=['nameOrig', 'newbalanceDest', 'oldbalanceDest', 'step', 'nameDest', 'isFlaggedFraud'])

# Filter the dataset to keep only 'TRANSFER' transactions
data = data[data['type'] == 'TRANSFER']

# Drop the 'type' column as it's no longer needed
data = data.drop(columns=['type'])

# Prepare data for training
X = data.drop(columns=['isFraud'])  # Features
y = data['isFraud']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(logreg_model, 'logreg_model.pkl')

# Evaluate the model
y_pred = logreg_model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:\n", accuracy_score(y_test, y_pred))

# Streamlit app for prediction
def predict_loan_default():
    st.title("Online Fraud Detection")
    st.write("Predict if a transaction is fraudulent or not.")

    # Collect user inputs
    amount_transferred = st.number_input("Enter the amount you transferred", min_value=0, max_value=10000, step=1, value=5000)
    oldbalance = st.number_input("Enter your Old Balance amount", min_value=0, step=1, value=10000)
    newbalance = st.number_input("Enter your New Balance amount", min_value=0, step=1, value=5000)

    if st.button("Submit"):
        # Collect User Input
        user_input = np.array([
            amount_transferred,
            oldbalance,
            newbalance
        ]).reshape(1, -1)

        print("User Input:", user_input)  # Print user input for debugging

        # Scale the user input
        user_input_scaled = scaler.transform(user_input)

        print("Scaled Input:", user_input_scaled)  # Print scaled input for debugging

        # Make prediction
        prediction = logreg_model.predict(user_input_scaled)

        print("Prediction:", prediction)  # Print prediction for debugging

        # Display prediction result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.write("Fraud Detected")
        else:
            st.write("No Fraud")

if __name__ == "__main__":
    predict_loan_default()
