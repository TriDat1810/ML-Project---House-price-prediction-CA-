import sys
import joblib
import pandas as pd


MODEL_PATH = "models/california_housing_model.joblib"

REQUIRED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


model = joblib.load(MODEL_PATH)

print("Model loaded successfully.\n")


if len(sys.argv) < 2:
    print("Usage:")
    print("python predict_csv.py <csv_file>")
    sys.exit(1)

file_path = sys.argv[1]


try:
    df = pd.read_csv(file_path)

except Exception as error:
    print(f"Could not read CSV file: {error}")
    sys.exit(1)


print(f"Loaded {len(df)} rows.")


missing_columns = [
    feature
    for feature in REQUIRED_FEATURES
    if feature not in df.columns
]


if missing_columns:

    print("\n❌ Missing required columns:")

    for column in missing_columns:
        print(f"- {column}")

    print("\nPrediction cannot continue.")

    sys.exit(1)


X = df[REQUIRED_FEATURES]


missing_values = X.isnull().sum()

if missing_values.any():

    print("\n⚠️ Missing values detected:")

    for feature, count in missing_values.items():

        if count > 0:
            print(f"- {feature}: {count} row(s)")

    print(
        "\nPrediction cannot continue yet."
    )

    print(
        "Missing-value handling will be added "
        "in the next step."
    )

    sys.exit(1)


predictions = model.predict(X)


results = df.copy()

results["PredictedHouseValue"] = predictions * 100000


print("\nPrediction results:\n")

print(
    results[
        REQUIRED_FEATURES + ["PredictedHouseValue"]
    ].to_string(index=False)
)