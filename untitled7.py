# Save this code as 'app.py'

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_FILE = 'trained_model.joblib'
# 
# !!! IMPORTANT !!! 
# YOU MUST UPDATE THIS LIST to exactly match the features (column names) 
# your trained_model.joblib was trained on, and in the correct order.
# 
FEATURE_NAMES = [
    'Delivery_Distance_km',
    'Time_of_Day_hours',
    'Num_Items',
    # Add any other features (e.g., 'Is_Rush_Hour', 'Weather_Condition_Encoded') here!
]
PREDICTION_TARGET = "Delivery Delay (minutes)"

# --- Load Model (Caching for Efficiency) ---
# st.cache_resource is the recommended way to load a model once
@st.cache_resource
def load_model():
    """Load the machine learning model using joblib."""
    try:
        # This is where the version error occurs in your environment
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        # Display the version error to the user
        st.error(f"Error loading model: {e}")
        st.caption("This usually means there is a mismatch in your 'scikit-learn' version.")
        st.caption("Try running: `pip install scikit-learn==1.3.2` in your terminal, or try updating it: `pip install -U scikit-learn`.")
        return None

# --- User Input Function ---
def user_input_features():
    """Collects user input using Streamlit widgets."""
    st.sidebar.header('Order Characteristics')

    # Example Input 1: Numerical slider
    distance = st.sidebar.slider(
        'Delivery Distance (km)',
        min_value=1.0, max_value=50.0, value=10.0, step=0.5
    )

    # Example Input 2: Numerical slider for time
    time_of_day = st.sidebar.slider(
        'Order Time (24h format, e.g., 14.5 is 2:30 PM)',
        min_value=0.0, max_value=23.99, value=18.0, step=0.01
    )

    # Example Input 3: Integer number input
    num_items = st.sidebar.number_input(
        'Number of Items in Order',
        min_value=1, max_value=50, value=3
    )

    # Dictionary of collected feature values
    data = {
        'Delivery_Distance_km': distance,
        'Time_of_Day_hours': time_of_day,
        'Num_Items': num_items,
        # IMPORTANT: If you added more features to FEATURE_NAMES, 
        # ensure they are collected here!
    }
    
    # Convert to DataFrame and ensure correct column order
    features = pd.DataFrame(data, index=[0])
    
    try:
        # Re-indexing to match the exact order expected by the model
        features = features[FEATURE_NAMES]
    except KeyError:
        st.error("Feature mismatch: Check that the keys in your 'data' dictionary match the 'FEATURE_NAMES' list.")
        return None
        
    return features


# --- Main Application ---
def main():
    st.set_page_config(page_title="Delivery Delay Prediction App")
    st.title("📦 Delivery Delay Prediction")
    st.markdown("Predicting delivery delay time in minutes.")

    # 1. Load the model
    model = load_model()

    if model is not None:
        # 2. Get user input
        input_df = user_input_features()
        
        if input_df is not None:
            st.subheader('Input Features Used for Prediction')
            st.dataframe(input_df, hide_index=True)

            # 3. Prediction
            if st.button('Predict Delay'):
                with st.spinner('Calculating prediction...'):
                    # The model.predict() call
                    prediction = model.predict(input_df)
                    
                    # Convert prediction to a readable format
                    predicted_delay = np.round(prediction[0], 2)
                    
                    # 4. Display the result
                    st.success('Prediction Complete!')
                    st.subheader(f'Predicted Result: {PREDICTION_TARGET}')
                    
                    # Display the prediction with context
                    if predicted_delay <= 0:
                        st.metric(label=PREDICTION_TARGET, value="0.00 minutes (Likely On Time)", delta_color="normal")
                        st.balloons()
                    else:
                        st.metric(label=PREDICTION_TARGET, value=f"{predicted_delay} minutes", delta=f"{predicted_delay} min", delta_color="inverse")
                        st.warning(f"Potential delay of {predicted_delay} minutes.")

if __name__ == '__main__':
    main()