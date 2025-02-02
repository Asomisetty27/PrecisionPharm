import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset (simulated data for now)
def load_data():
    data = {
        'CYP2C19': np.random.choice(['Poor', 'Intermediate', 'Extensive', 'Ultra-rapid'], 500),
        'CYP3A4': np.random.choice(['Low', 'Normal', 'High'], 500),
        'Liver_Enzyme_ALT': np.random.uniform(10, 100, 500),
        'Liver_Enzyme_AST': np.random.uniform(10, 80, 500),
        'Kidney_Function_Creatinine': np.random.uniform(0.5, 1.5, 500),
        'Drug_Metabolism_Rate': np.random.uniform(0.1, 2.5, 500)
    }
    return pd.DataFrame(data)

# Preprocess data
def preprocess_data(df):
    label_encoders = {}
    for col in ['CYP2C19', 'CYP3A4']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    scaler = StandardScaler()
    X = df.drop(columns=['Drug_Metabolism_Rate'])
    y = df['Drug_Metabolism_Rate']
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return X_scaled, y

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    print("Data preprocessing completed successfully.")
