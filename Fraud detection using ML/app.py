# loan_default_app.py
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and preprocess data
data = pd.read_csv("onlinefraud.csv")
data=data.drop(columns=['nameOrig'])
data=data.drop(columns=['newbalanceDest'])
data=data.drop(columns=['oldbalanceDest'])
data=data.drop(columns=['step'])
data=data.drop(columns=['nameDest'])
data=data.drop(columns=['isFlaggedFraud'])

# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder

cols_to_label_encode = ['type']
label_encoder = LabelEncoder()

for col in cols_to_label_encode:
    data[col] = label_encoder.fit_transform(data[col])

# Prepare data for training
X = data.drop(columns=['isFraud'])  # Features
y = data['isFraud'] 

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(logreg_model, 'logreg_model.pkl')

# Streamlit app for prediction
def predict_loan_default():
    st.header("Online Fraud Detection")
    image = Image.open('loan_image.jpeg')
    #st.image(image, caption='Loan Default Prediction', use_column_width=True)

    # Collect user inputs
    

    transaction = st.selectbox("Enter your Transaction Tyoe", ("Cash Out", "Debit", "Payment", "Transfer"))
    transaction_dict = {"Cash Out": 0, "Debit": 1, "Payment": 2, "Transfer": 3}
    transaction_input =  transaction_dict [transaction]
    amount_transfered= st.number_input("Enter your amount you transferred", min_value=0, max_value=10000, step=1)
    oldbalance= st.number_input("Enter your Old Balance amount", min_value=0, step=1)
    newbalance= st.number_input("Enter your New Balance amount", min_value=0, step=1)
      
    if st.button("Submit"):
        # Step 2: Collect User Input
        user_input = np.array([
           transaction_input,
           amount_transfered,
           oldbalance,
           newbalance
        ]).reshape(1, -1)

        # Scale the user input
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = logreg_model.predict(user_input_scaled)

        # Display prediction result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.write("Fraud Detected")
        else:
            st.write("No Fraud")

if __name__ == "__main__":
    predict_loan_default()