import joblib
import pandas as pd
import tensorflow as tf

NUMERIC_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

BOOLEAN_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

CATEGORICAL_COLUMNS = [
    "furnishingstatus"
]

TARGET_COLUMN = "price"
model = tf.keras.models.load_model("models/houseprice_model.keras")
preprocessor = joblib.load("models/houseprice_preprocessor.pkl")

def predict(
         area,
         bedrooms,
         bathrooms,
         stories,
         mainroad,
         guestroom,
         basement,
         hotwaterheating,
         airconditioning,
         parking,
         prefarea,
        furnishingstatus
):
    house = pd.DataFrame([
    {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
      }
   ])
    
    processed = preprocessor.transform(house)

    prediction = model.predict(processed, verbose=0)

    return float(prediction[0][0])

if __name__ == "__main__":
    house_price = predict (
        area = 7420,
         bedrooms = 4,
         bathrooms = 2,
         stories = 3,
         mainroad = "yes",
         guestroom ="no",
         basement = "no",
         hotwaterheating = "no",
         airconditioning = "yes",
         parking = 1,
         prefarea = "yes",
         furnishingstatus = "furnished"
    )    

    print(f"Predicted Price: {house_price:.2f}")

    
  