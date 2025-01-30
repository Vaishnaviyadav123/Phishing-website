import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('phishing_website_data.csv')

# Basic preprocessing
df['url_length'] = df['URL'].apply(lambda x: len(x))
df['num_digits'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in x))
df['num_special_chars'] = df['URL'].apply(lambda x: sum(not c.isalnum() for c in x))

# Prepare features and labels
X = df[['url_length', 'num_digits', 'num_special_chars']]
y = df['Label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Random Forest as an example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model
joblib.dump(clf, 'phishing_detector_model.pkl')
