from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_models(models, X_test, y_test):
    print("\n[MODULE 7] MODEL EVALUATION")

    rows = []

    for name, model in models.items():
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        rows.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"\n{name}")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

    results = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    results.to_csv("outputs/model_results.csv", index=False)

    return results

def select_best_model(results, models):
    # Primary criterion: lowest RMSE
    best_row = results.loc[results["RMSE"].idxmin()]
    best_name = best_row["Model"]
    return best_name, models[best_name]
