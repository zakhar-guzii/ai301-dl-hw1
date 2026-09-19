import numpy as np

# ADC grid of the pressure sensor
P_MIN = -1.895744294564641
P_MAX = 64.8209917386395
P_STEP = (P_MAX - P_MIN) / 949


def snap(pred) -> np.ndarray:
    pred = np.clip(np.asarray(pred), P_MIN, P_MAX)
    return P_MIN + np.round((pred - P_MIN) / P_STEP) * P_STEP
