# predictor.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import build_lstm

def forecast_missed(ts, lookback=6):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[['missed']])

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = build_lstm((X.shape[1], 1))
    model.fit(X, y, epochs=10, verbose=0)

    return model
