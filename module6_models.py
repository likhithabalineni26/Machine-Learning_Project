from sklearn.linear_model import Ridge, Lasso, ElasticNet

def train_models(X_train, y_train):
    print("\n[MODULE 6] MODEL BUILDING")

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Elastic Net Regression": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained successfully.")

    return models
