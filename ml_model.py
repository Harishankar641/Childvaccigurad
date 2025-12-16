# ml_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    df = df.copy()
    df["missed"] = (df["status"] == "missed").astype(int)

    X = pd.get_dummies(
        df[["age", "dose_number", "state", "provider_type"]],
        drop_first=True
    )
    y = df["missed"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    # ✅ SAVE MODEL + FEATURE NAMES
    with open("models/missed_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": X.columns.tolist()
            },
            f
        )
