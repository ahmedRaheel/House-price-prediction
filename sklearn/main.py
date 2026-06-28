import pandas as pd 
import joblib

model = joblib.load("models/sklearn_housing_model.pkl")

sample = pd.DataFrame([
    {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 1,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }
])

prediction = model.predict(sample)

print(f"Predicted Price: {prediction[0]:.2f}")