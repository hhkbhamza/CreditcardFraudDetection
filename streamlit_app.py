import streamlit as st
import joblib
import pandas as pd
from openai import OpenAI
import sys
print("Python executable in use:", sys.executable)


api_key = st.secrets["API_KEY"]

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Load models
voting_model = joblib.load('voting_clf.pkl')
xgb_model = joblib.load('xgb_model.pkl')
xgb_smote_model = joblib.load('xgboost-SMOTE.pkl')

# Load feature template
feature_template = pd.read_pickle("features_template.pkl")  # Ensure this file is in the same directory

def predict_with_all_models(input_data):
    """Get predictions and probabilities from each model."""
    results = {}
    
    results['Voting Model'] = {
        "prediction": voting_model.predict(input_data)[0],
        "probability": voting_model.predict_proba(input_data)[0][1]  # Probability for class '1' (fraud)
    }
    
    results['XGBoost'] = {
        "prediction": xgb_model.predict(input_data)[0],
        "probability": xgb_model.predict_proba(input_data)[0][1]
    }
    
    results['XGBoost with SMOTE'] = {
        "prediction": xgb_smote_model.predict(input_data)[0],
        "probability": xgb_smote_model.predict_proba(input_data)[0][1]
    }
    
    return results

def explain_prediction(probability, input_data, threshold=0.7):
    """Generate an explanation of the prediction."""
    # Structure the input data for display
    input_dict = input_data.iloc[0].to_dict()
    
    # Customize explanation prompt
    prompt = f"""
    You are a data scientist tasked with explaining a credit card fraud prediction.

    The model has predicted a {round(probability * 100, 1)}% probability of fraud for a transaction based on the following details:
    {input_dict}

    Top features contributing to fraud detection:

    | Feature               | Importance     |
    |-----------------------|----------------|
    | Transaction Amount    | 0.40           |
    | City Population       | 0.25           |
    | Merchant Latitude     | 0.15           |
    | Merchant Longitude    | 0.10           |
    | Category              | 0.07           |
    | State                 | 0.03           |

    Summary:
    - If the fraud probability is above {threshold*100}%, provide a 3-sentence explanation of why it is likely fraudulent.
    - If the fraud probability is below {threshold*100}%, provide a 3-sentence explanation of why it may not be fraudulent.

    Avoid mentioning the machine learning model directly; explain the prediction based on the transaction details.
    """
    
    # Get explanation from OpenAI API
    response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def ensemble_predict_with_threshold(input_data, threshold=0.7):
    """Calculate ensemble prediction with a threshold."""
    individual_results = predict_with_all_models(input_data)
    predictions = [1 if result["probability"] > threshold else 0 for result in individual_results.values()]
    # Use majority voting on the adjusted predictions
    return 1 if predictions.count(1) > predictions.count(0) else 0, individual_results

def main():
    st.title("Credit Card Fraud Detection with Multiple Models")
    st.write("Enter the transaction details below to check if it's potentially fraudulent.")
    
    with st.form("fraud_detection_form"):
        # Collect inputs for the most important features
        amt = st.number_input("Transaction Amount ($)", min_value=0.0)
        city_pop = st.number_input("City Population", min_value=0)
        merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0)
        merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0)
        
        category = st.selectbox("Transaction Category", [
            "grocery_net", "kids_pets", "travel", "grocery_pos", "misc_net", "gas_transport", "entertainment"
        ])
        
        state = st.selectbox("State", ["TX", "NY", "CA", "FL", "other"])
        
        # Fraud probability threshold slider
        threshold = st.slider("Set fraud probability threshold", min_value=0.5, max_value=1.0, step=0.05, value=0.7)
        
        # Submit button to run the predictions
        submit_button = st.form_submit_button("Check for Fraud")

    # Only proceed if the form is submitted
    if submit_button:
        # Prepare data for prediction
        data = {
            'amt': [amt],
            'city_pop': [city_pop],
            'merch_lat': [merch_lat],
            'merch_long': [merch_long],
            f'category_{category}': [1],
        }

        # Add dummy columns for all categories and states to match model's input shape
        categories = ["grocery_net", "kids_pets", "travel", "grocery_pos", "misc_net", "gas_transport", "entertainment"]
        for cat in categories:
            if f'category_{cat}' not in data:
                data[f'category_{cat}'] = [0]
        
        states = ["TX", "NY", "CA", "FL", "other"]
        for s in states:
            data[f'state_{s}'] = [1 if s == state else 0]

        input_df = pd.DataFrame(data)

        # Align input_df with the feature template
        input_df = input_df.reindex(columns=feature_template.columns, fill_value=0)

        # Get predictions and probabilities from all models with the adjusted threshold
        ensemble_prediction, individual_results = ensemble_predict_with_threshold(input_df, threshold=threshold)
        
        # Display each model's prediction and confidence score
        st.subheader("Individual Model Predictions, Recall Scores, and Confidence, and Explanation")
        for model_name, result in individual_results.items():
            result_text = "Fraud" if result["prediction"] == 1 else "Legitimate"
            confidence = round(result["probability"] * 100, 2) # Convert to percentage
            explanation = explain_prediction(result["probability"], input_df, threshold)
            st.write(f"{model_name}: {result_text} (Confidence: {confidence}%)")
            st.write("Explanation:", explanation)
        
        # Display ensemble prediction result
        st.subheader("Ensemble Prediction")
        if ensemble_prediction == 1:
            st.error("Warning: The ensemble prediction suggests this transaction is likely fraudulent.")
        else:
            st.success("The ensemble prediction suggests this transaction appears to be legitimate.")

# Run the app
if __name__ == "__main__":
    main()
