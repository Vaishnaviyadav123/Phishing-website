import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
