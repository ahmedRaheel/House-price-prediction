import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import Dataset
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
class HousePriceDataset(Dataset):

    def __init__(self, csv_path):
        
        df = pd.read_csv(csv_path)

        self.X = df[NUMERIC_COLUMNS + BOOLEAN_COLUMNS + CATEGORICAL_COLUMNS]
        self.y = df[[TARGET_COLUMN]]

        self.preprocessor  = ColumnTransformer( transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS), 
            ("bool", OneHotEncoder(drop="if_binary"), BOOLEAN_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ])

        X_processed = self.preprocessor.fit_transform(self.X)
        self.X = torch.tensor(X_processed, dtype=torch.float)
        self.y = torch.tensor(self.y.values, dtype= torch.float)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


