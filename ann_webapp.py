from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import pandas as pd

# Load model
model = load_model('trained_model.h5')

# Load pickle files
def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error! File not found: {filepath}")
        return None

scaler = load_pickle_file("dataScaler.pkl")
gender_encoder = load_pickle_file("gender_encoder.pkl")
geo_encoder = load_pickle_file("geography_encoder.pkl")

#streamlit app headrer
st.title('Customer Churn Prediction')

#display breif description
st.markdown("""
### 🔍 Customer Churn Prediction

This app uses a **trained deep learning model** to predict the probability that a customer will churn based on their profile.  
The prediction is made using the following process:

1. **Input Features**:  
   The model takes into account features like:
   - Credit Score  
   - Geography  
   - Gender  
   - Age  
   - Tenure with the bank  
   - Account Balance  
   - Number of Products  
   - Credit Card ownership  
   - Active Membership status  
   - Estimated Salary  

2. **Preprocessing**:
   - Gender is **label encoded**.
   - Geography is **one-hot encoded**.
   - Numerical features are **scaled** using a pre-trained scaler.

3. **Prediction**:
   - The preprocessed data is fed into a pre-trained neural network model (`trained_model.h5`).
   - The model outputs a **churn probability score** between 0 and 1.

4. **Result Interpretation**:
   - If the churn probability is **greater than 0.5**, the customer is **likely to churn**.
   - Otherwise, the customer is **not likely to churn**.

Upload your data or input it manually to see real-time predictions.  
The model helps banks understand and **proactively manage customer retention**.

---
""")

#collect user inputs
credit_score = st.number_input("Enter Credit Score:", min_value=100, max_value=850)
geography = st.selectbox('Select Geography:', geo_encoder.categories_[0])
gender = st.selectbox('Select Gender:', gender_encoder.classes_)
age = st.slider('Select Age:', 18, 92)
tenure = st.slider('Select Tenure (Years):', 0, 10)
balance = st.number_input('Enter Account Balance:', min_value=0.0)
num_of_products = st.slider('Number of Products:', 1, 4)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])
estimated_salary = st.number_input('Enter Estimated Salary:', min_value=0.0)

#button to make prediction
if st.button("Predict Churn Probability"):
    #validate the inputs
    if credit_score <= 0 or estimated_salary <= 0 or balance < 0:
        st.error("Please enter valid values for Credit Score, Balance, and Estimated Salary.")
    else:
        #prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [gender_encoder.transform([gender])[0]], 
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        # One-hot encode Geography
        geo_encoded = geo_encoder.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

        # Combine the input data with the encoded geography
        input_data = pd.concat((input_data.reset_index(drop=True), geo_encoded_df), axis=1)

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)

        #predict churn probability
        prediction=model.predict(input_data_scaled)
        prediction_proba=prediction[0][0]

        # Display the results
        st.write(f'Churn Probability: {prediction_proba:.2f}')

        # Display likelihood of churn
        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')

        st.progress(int(prediction_proba*100))