import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load pre-trained model and preprocessing tools
def load_model():
    model = joblib.load('drug_metabolism_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

# Function to predict drug metabolism rate for new patients
def predict_metabolism(genetic_cyp2c19, genetic_cyp3a4, liver_alt, liver_ast, creatinine):
    model, scaler, label_encoders = load_model()
    
    try:
        input_data = np.array([[
            label_encoders['CYP2C19'].transform([genetic_cyp2c19])[0],
            label_encoders['CYP3A4'].transform([genetic_cyp3a4])[0],
            liver_alt, liver_ast, creatinine
        ]])
    except ValueError as e:
        return f"Error: {e}. Check input values."
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return prediction

if __name__ == "__main__":
    # Example usage
    result = predict_metabolism('Extensive', 'Normal', 35.2, 25.1, 1.0)
    print("Predicted Drug Metabolism Rate:", result)
