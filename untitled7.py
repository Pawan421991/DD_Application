# Save this code as 'app.py'

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_FILE = 'trained_model.joblib'

# !!! IMPORTANT: YOU MUST UPDATE THIS LIST !!!
# Replace these with the EXACT feature names (column headers) 
# your model was trained on, in the correct order.
FEATURE_NAMES = [
    'Delivery_Distance_km',
    'Time_of_Day_hours',
    'Num_Items',
    # Example placeholders: 'Is_Weekend', 'Traffic_Density_Score', 'Delivery_Area_Encoded', ...
    # Add ALL your model's required input features here.
]
PREDICTION_TARGET = "Delivery Delay (minutes)"

# --- Load Model (Caching for Efficiency) ---
# st.cache_resource is used to load the model only once, speeding up the app.
@st.cache_resource
def load_model():
    """Load the machine learning model using joblib."""
    try:
        # This line will now succeed after fixing your scikit-learn version
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory as this script.")
        return None
    except Exception as e:
        # This is the error we are fixing with the terminal commands
        st.error(f"Error loading model: {e}")
        st.caption("This error means your 'scikit-learn' version is incompatible with the model file.")
        st.caption("Fix this by running the appropriate `pip install scikit-learn==X.Y.Z` command in your terminal.")
        return None

# --- User Input Function ---
def user_input_features():
    """Collects user input using Streamlit widgets for prediction."""
    st.sidebar.header('Input Order Characteristics')

    # Example Input 1: Delivery Distance
    distance = st.sidebar.slider(
        'Delivery Distance (km)',
        min_value=0.5, max_value=50.0, value=10.0, step=0.1
    )

    # Example Input 2: Time of Day (24-hour format)
    time_of_day = st.sidebar.slider(
        'Order Time (Hours - 0.0 to 23.99)',
        min_value=0.0, max_value=23.99, value=18.0, step=0.01
    )

    # Example Input 3: Number of Items
    num_items = st.sidebar.number_input(
        'Number of Items in Order',
        min_value=1, max_value=50, value=3
    )

    # Dictionary containing the input values
    data = {
        'Delivery_Distance_km': distance,
        'Time_of_Day_hours': time_of_day,
        'Num_Items': num_items,
        # Add values for any other features you added to FEATURE_NAMES here
    }
    
    # Convert to DataFrame and ensure correct column order
    features = pd.DataFrame(data, index=[0])
    
    try:
        # CRUCIAL: Re-indexing to match the exact column order expected by the model
        features = features[FEATURE_NAMES]
    except KeyError:
        st.error("Input Feature Mismatch: Please ensure the features collected above match the 'FEATURE_NAMES' list.")
        return None
        
    return features


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Delivery Delay Prediction App")
    st.title("📦 Delivery Delay Prediction")
    st.markdown("Use the sidebar to adjust input values and predict the time taken for delivery.")

    # 1. Load the model
    model = load_model()

    if model is not None:
        # 2. Get user input
        input_df = user_input_features()
        
        if input_df is not None:
            st.subheader('Input Data')
            st.dataframe(input_df, hide_index=True)

            # 3. Prediction
            if st.button('Predict Delay'):
                with st.spinner('Calculating prediction...'):
                    
                    # Make the prediction
                    prediction = model.predict(input_df)
                    
                    # Process the result (e.g., round to 2 decimal places)
                    predicted_delay = np.round(prediction[0], 2)
                    
                    # 4. Display the result
                    st.success('Prediction Complete!')
                    st.subheader(f'Predicted {PREDICTION_TARGET}:')
                    
                    if predicted_delay <= 0:
                        st.metric(label=PREDICTION_TARGET, value="0.00 minutes (On Time)", delta_color="normal")
                        st.balloons()
                    else:
                        st.metric(label=PREDICTION_TARGET, value=f"{predicted_delay} minutes", delta=f"{predicted_delay} min", delta_color="inverse")
                        st.warning(f"The model predicts a delay of {predicted_delay} minutes.")

if __name__ == '__main__':
    main()