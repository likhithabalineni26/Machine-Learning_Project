from module1_dataset import load_dataset
from module2_understanding import understand_dataset
from module3_preprocessing import preprocess_data
from module4_eda import create_eda_plots
from module5_features import create_features
from module6_models import train_models
from module7_evaluation import evaluate_models, select_best_model
from module8_save_model import save_model

def main():
    print("\n" + "=" * 65)
    print(" CROP YIELD PREDICTION USING RIDGE, LASSO AND ELASTIC NET")
    print("=" * 65)

    df = load_dataset("data/crop_yield.csv")
    understand_dataset(df)

    df = create_features(df)
    create_eda_plots(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)
    best_name, best_model = select_best_model(results, models)

    save_model(best_name, best_model, preprocessor)

    print("\n" + "=" * 65)
    print(f"BEST MODEL: {best_name}")
    print("=" * 65)
    print("Project completed successfully.")
    print("Check outputs/ for evaluation results.")
    print("Check models/ for the saved best model.")
    print("Check plots/ for EDA graphs.")

if __name__ == "__main__":
    main()
