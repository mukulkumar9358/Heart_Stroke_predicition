import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('heart.csv')
print(df.columns)

# Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

# Features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model using joblib
joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
