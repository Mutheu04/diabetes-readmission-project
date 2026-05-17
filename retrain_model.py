import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Load processed data
df = pd.read_csv('data/processed/diabetic_processed.csv')

# Split
X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
scaler.fit(X_train)

# Train
sample_weights = compute_sample_weight('balanced', y_train)
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    min_samples_leaf=20, random_state=42
)
gb_model.fit(X_train, y_train, sample_weight=sample_weights)

# Save
joblib.dump(gb_model, 'outputs/gb_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
joblib.dump(list(X.columns), 'outputs/feature_names.pkl')

print("Model retrained and saved with scikit-learn 1.8.0")