import  pandas as pd 
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing  import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

df = pd.read_csv("data/Housing.csv")

X = df[NUMERIC_COLUMNS + BOOLEAN_COLUMNS + CATEGORICAL_COLUMNS]
y = df[TARGET_COLUMN]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("bool", OneHotEncoder(drop="if_binary"), BOOLEAN_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")

joblib.dump(model, "models/sklearn_housing_model.pkl")
print("Scikit-learn model saved.")