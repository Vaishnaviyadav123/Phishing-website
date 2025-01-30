from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = joblib.load('phishing_detector_model.pkl')

# Function to preprocess URLs
def preprocess_url(url):
    parsed_url = urlparse(url)
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    return [url_length, num_digits, num_special_chars]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get incoming data
    url = data.get('url')

    if not url:
        return jsonify({'result': 'Error: No URL provided'}), 400

    try:
        # Preprocess the URL
        processed_url = preprocess_url(url)

        # Make prediction
        prediction = model.predict([processed_url])
        result = "Phishing" if prediction[0] == 1 else "Legitimate"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import requests # type: ignore
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# Feature extraction functions
def url_length(url):
    return len(url)

def has_https(url):
    return 1 if url.startswith('https') else 0

def has_ip_address(url):
    parsed_url = urlparse(url)
    return 1 if any(char.isdigit() for char in parsed_url.netloc) else 0

def num_of_subdomains(url):
    parsed_url = urlparse(url)
    return len(parsed_url.netloc.split('.')) - 1

def contains_keywords(url, keywords):
    return sum(1 for keyword in keywords if keyword in url)

# Load your dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Process data: extract features
def extract_features(df):
    features = []
    keywords = ['login', 'secure', 'bank', 'account', 'verify', 'user', 'confirm']
    
    for url in df['URL']:
        features.append([
            url_length(url),
            has_https(url),
            has_ip_address(url),
            num_of_subdomains(url),
            contains_keywords(url, keywords)
        ])
    
    return np.array(features)

# Train and test the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main pipeline
def phishing_detection_pipeline(file_path):
    # Load and prepare the data
    df = load_data(file_path)
    X = extract_features(df)
    y = df['Label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features (important for some algorithms)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

# Run the phishing detection pipeline
if __name__ == '__main__':
    # Replace 'phishing_data.csv' with the actual path to your dataset
    phishing_detection_pipeline('phishing_data.csv')
