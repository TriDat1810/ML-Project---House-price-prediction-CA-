import joblib
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# Load trained model
model = joblib.load("models/california_housing_model.joblib")

print("Model loaded successfully.")


# Load the same dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame


# Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]


# Recreate the same test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Predict using the saved model
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")