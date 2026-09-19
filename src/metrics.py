from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred, u_out):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    u_out = np.asarray(u_out).ravel()

    mask = u_out == 0

    me = (y_pred[mask]  - y_true[mask]).mean()
    mae = mean_absolute_error(y_pred=y_pred[mask], y_true=y_true[mask])
    rmse = root_mean_squared_error(y_pred=y_pred[mask], y_true=y_true[mask])
    r2 = r2_score(y_pred=y_pred[mask], y_true=y_true[mask])

    results = {'ME': me,
               "MAE": mae, 
               'RMSE': rmse,
               'R2': r2
    }
    return results