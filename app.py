import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model, selected features, and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('selected_features.pkl', 'rb') as feature_file:
    selected_features = pickle.load(feature_file)

# Initialize the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the Streamlit layout
st.title("Phishing Detection")

# Create two columns for the input boxes
col1, col2 = st.columns(2)

# Create input fields for the selected features in the two columns
if 'inputs' not in st.session_state:
    st.session_state.inputs = {feature: "" for feature in selected_features}  # Initialize empty inputs

with col1:
    for i, feature in enumerate(selected_features[:len(selected_features)//2]):
        st.session_state.inputs[feature] = st.text_input(f"Enter {feature}", value=st.session_state.inputs[feature])

with col2:
    for i, feature in enumerate(selected_features[len(selected_features)//2:]):
        st.session_state.inputs[feature] = st.text_input(f"Enter {feature}", value=st.session_state.inputs[feature])

# Add a submit button
if st.button("Predict"):
    # Process the inputs and convert them into a dataframe
    try:
        # Convert inputs to numeric values
        input_data = {feature: [float(st.session_state.inputs[feature]) if st.session_state.inputs[feature] else np.nan] for feature in selected_features}
        input_df = pd.DataFrame(input_data)

        # Handle missing or invalid inputs
        if input_df.isnull().values.any():
            st.error("⚠️ Please enter valid numeric values for all fields.")
        else:
            # Scaling the input features using the same scaler
            input_scaled = scaler.transform(input_df)

            # Predict using the loaded model
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)[0][1]

            # Display the prediction
            st.write(f"Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
            st.write(f"Prediction probability: {proba * 100:.2f}%")
    except ValueError as e:
        st.error("⚠️ Please enter valid numeric values for all fields.")
        print(f"Error: {e}")
