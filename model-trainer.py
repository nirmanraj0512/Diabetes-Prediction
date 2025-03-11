# model-trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Create Models directory if not exists
os.makedirs('Models', exist_ok=True)

def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('Dataset/diabetes.csv')
    
    # Replace zeros with NaN
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_columns] = df[zero_columns].replace(0, np.nan)
    
    # Fill missing values with mean
    df.fillna(df.mean(), inplace=True)
    
    return df

def train_model():
    """Main training pipeline"""
    # Load and preprocess data
    dataset = load_data()
    
    # Feature selection (as used in the notebook)
    X = dataset[['Glucose', 'Insulin', 'BMI', 'Age']]
    y = dataset['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model (parameters from notebook)
    knn = KNeighborsClassifier(
        n_neighbors=24,
        metric='minkowski',
        p=2
    )
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = knn.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    pickle.dump(knn, open('Models/model.pkl', 'wb'))
    pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))
    print("\nModel and scaler saved to Models/ directory")

if __name__ == '__main__':
    train_model()