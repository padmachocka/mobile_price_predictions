import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

def run_pipeline():
    # Load
    df = pd.read_csv("train.csv")

    # Safe feature engineering (no target leakage here)
    df['sc_w'] = df['sc_w'].replace(0, 0.1)
    df['mobile_wt'] = df['mobile_wt'].replace(0, 0.1)
    df['screen_area'] = df['sc_h'] * df['sc_w']
    df['battery_per_weight'] = df['battery_power'] / df['mobile_wt']

    # Proper PPI
    diag_cm = np.sqrt(df['sc_h']**2 + df['sc_w']**2)
    diag_in = diag_cm / 2.54
    ppi = np.sqrt(df['px_height']**2 + df['px_width']**2) / diag_in.replace(0, np.nan)
    df['ppi'] = ppi.fillna(ppi.median())

    # Split
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Columns to scale (fit ONLY on train inside the pipeline)
    scale_cols = ['battery_power', 'ram', 'int_memory', 'mobile_wt',
                  'screen_area', 'battery_per_weight', 'ppi']

    preproc = ColumnTransformer(
        transformers=[('scale', StandardScaler(), scale_cols)],
        remainder='passthrough'
    )

    # Model
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[('preproc', preproc), ('model', xgb)])

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 4, 5, 6],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=20, cv=3, scoring='accuracy',
        verbose=1, n_jobs=-1, random_state=42
    )

    # MLflow
    mlflow.set_tracking_uri("http://18.175.251.68:5000")
    mlflow.set_experiment("xgb_random_search_experiment")
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="xgb_random_search") as run:
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        preds = best_pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("test_accuracy", float(acc))

        print("Best Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, preds))

        # Save ONE artifact that includes preprocessing + model
        with open("best_xgb_pipeline.pkl", "wb") as f:
            pickle.dump(best_pipe, f)

        # Optional: explicit model logging (autolog already does this)
        signature = infer_signature(
            X_train.iloc[:5],  # raw features
            best_pipe.predict_proba(X_train.iloc[:5])
        )
        # If you keep manual log, consider disabling autolog model logging to avoid duplicates.
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

if __name__ == "__main__":
    run_pipeline()
