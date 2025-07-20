import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop rows with missing BMI
df = df.dropna(subset=["bmi"])

# Drop 'id' column
df = df.drop("id", axis=1)

# Encode categorical columns
label_encoders = {}
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature/target split
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Scale numeric features
scaler = StandardScaler()
X[["age", "avg_glucose_level", "bmi"]] = scaler.fit_transform(X[["age", "avg_glucose_level", "bmi"]])

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model, encoders, scaler
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

# Optional: Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
