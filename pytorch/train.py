import torch.nn  as nn
import torch
from torch.utils.data import DataLoader

from model import HousePriceModel
from dataset import HousePriceDataset
from model_manager import ModelManager

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

def train():
    data = HousePriceDataset("data/housing.csv")
    preprocessor = data.preprocessor
    ModelManager.save_preprocessor(preprocessor)


    loader = DataLoader(data, batch_size=16)
    
    input_size = data.X.shape[1]
    model = HousePriceModel(input_size)

    lose_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300

    for epoch in range(300):
        model.train()
        total_loss =0

        for features, targets in loader:
            predictions = model(features)

            loss = lose_fn(features, predictions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
            if epoch % 100 == 0 and epoch > 0:
                ModelManager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    input_size= input_size,
                    loss=total_loss
                )
    
    ModelManager.save_model(model)
    ModelManager.save_preprocessor(data.preprocessor)
    print("Training completed.")

if __name__ == "__main__":
    train()
