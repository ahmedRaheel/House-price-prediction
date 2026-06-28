import pandas as pd 
import torch 

from model_manager import ModelManager

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
    preprocessor = ModelManager.load_preprocessor()
    

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
    
    X = preprocessor.transform(house)
    X = torch.tensor(
        X,
        dtype=torch.float32
    )
    input_size = X.shape[1]
    model = ModelManager.load_model(input_size)

    model.eval()

    with torch.no_grad():
        prediction = model(X)

    return prediction.item()

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

