import joblib
import pandas as pd


model = joblib.load("models/california_housing_model.joblib")

print("Model loaded successfully.")

feature_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_,).sort_values(ascending=False)

def importance_level(value):
    if value >= 0.20:
        return "HIGH"
    elif value >= 0.05:
        return "MEDIUM"
    else:
        return "LOW"

def get_number(feature_name, min_value=None, max_value=None):

    while True:

        value = input(f"{feature_name}: ").strip()

        # Allow missing value
        if value == "":
            return None

        try:
            value = float(value)

        except ValueError:
            print(
                "Invalid input. Please enter a number "
                "or press Enter to leave it empty."
            )
            continue

        if min_value is not None and value < min_value:
            print(
                f"Value must be at least {min_value}."
            )
            continue

        if max_value is not None and value > max_value:
            print(
                f"Value must be at most {max_value}."
            )
            continue

        return value


med_inc = get_number(
    "MedInc",
    min_value=0
)

house_age = get_number(
    "HouseAge",
    min_value=0
)

ave_rooms = get_number(
    "AveRooms",
    min_value=0
)

ave_bedrms = get_number(
    "AveBedrms",
    min_value=0
)

population = get_number(
    "Population",
    min_value=0
)

ave_occup = get_number(
    "AveOccup",
    min_value=0
)

latitude = get_number(
    "Latitude",
    min_value=-90,
    max_value=90
)

longitude = get_number(
    "Longitude",
    min_value=-180,
    max_value=180
)

house = pd.DataFrame([{
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
}])

missing_features = house.columns[house.isnull().any()].tolist()

if missing_features:

    print("\n⚠️ WARNING")
    print("Some information is missing.\n")

    print("Missing features:")

    for feature in missing_features:

        importance = feature_importance[feature]
        level = importance_level(importance)

        print(
            f"- {feature}: "
            f"{level} importance "
            f"({importance:.3f})"
        )

    print(
        "\nThe model will NOT make a prediction "
        "with missing information yet."
    )

    print(
        "Please provide all required information."
    )

else:
    
    prediction = model.predict(house)[0]

    print(
        f"\nPredicted house value: "
        f"${prediction * 100000:,.2f}"
    )