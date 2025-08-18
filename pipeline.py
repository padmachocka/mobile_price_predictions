import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

def run_pipeline():
    # Load data
    df = pd.read_csv("train.csv")

    # Feature engineering
    df['sc_w'] = df['sc_w'].replace(0, 0.1)
    df['mobile_wt'] = df['mobile_wt'].replace(0, 0.1)
    df['screen_area'] = df['sc_h'] * df['sc_w']
    df['battery_per_weight'] = df['battery_power'] / df['mobile_wt']
    df['ppi'] = np.sqrt(df['px_height']*2 + df['px_width']*2) / df['sc_w']

    scale_cols = ['battery_power', 'ram', 'int_memory', 'mobile_wt',
                  'screen_area', 'battery_per_weight', 'ppi']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Split
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # MLflow experiment
    mlflow.set_tracking_uri("http://18.175.251.68:5000")
    mlflow.set_experiment("xgb_random_search_experiment")
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_random = RandomizedSearchCV(
        xgb_model, param_distributions=param_grid,
        n_iter=20, cv=3, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42
    )

    with mlflow.start_run(run_name="xgb_random_search") as run:
        xgb_random.fit(X_train, y_train)
        best_xgb = xgb_random.best_estimator_
        preds = best_xgb.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("test_accuracy", float(acc))
        print("Best Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, preds))

        # Save model to pickle
        with open("best_xgb_model.pkl", "wb") as f:
            pickle.dump(best_xgb, f)

        # Log model in MLflow
        signature = infer_signature(X_train[:5], best_xgb.predict_proba(X_train[:5]))
        mlflow.xgboost.log_model(best_xgb, artifact_path="model",
                                 signature=signature, input_example=X_train[:5])

if __name__ == "__main__":
    run_pipeline()
