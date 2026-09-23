"""
Train the KNN model on the synthetic demo data.

The original model was trained on the private hackathon dataset. A KNN model
stores its training data inside the .pkl file, so the original model is NOT
published. This script rebuilds an equivalent model from the demo data.
"""
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("demo_data.csv")
df["Work load"] = df["Work load"].astype(str).str.replace(",", "").astype(int)

X = df.drop(columns=["ID", "Absenteeism time in hours"])
y = df["Absenteeism time in hours"]

# Same hyperparameters found by GridSearchCV on the original data
model = KNeighborsClassifier(n_neighbors=3, weights="distance", leaf_size=10)
model.fit(X, y)

joblib.dump(model, "knn_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("Demo model saved. (Accuracy on random demo data is not meaningful.)")
