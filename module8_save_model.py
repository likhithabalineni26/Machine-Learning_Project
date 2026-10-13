from pathlib import Path
import joblib

def save_model(best_name, model, preprocessor):
    print("\n[MODULE 8] SAVE FINAL MODEL")

    Path("models").mkdir(exist_ok=True)

    package = {
        "model_name": best_name,
        "preprocessor": preprocessor,
        "model": model
    }

    joblib.dump(package, "models/best_crop_yield_model.pkl")
    print("Saved: models/best_crop_yield_model.pkl")
