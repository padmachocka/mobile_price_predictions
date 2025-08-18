#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('curl -s http://checkip.amazonaws.com')


# In[4]:


# Basic imports
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings('ignore')

# Load the training data
df = pd.read_csv('train.csv')

# Preview the dataset
print("📊 Shape of dataset:", df.shape)
df.head()
df.describe()


# In[5]:


df.info()


# In[6]:


# Make a copy to avoid changing original
df_fe = df.copy()

# Avoid divide-by-zero errors
df_fe['sc_w'] = df_fe['sc_w'].replace(0, 0.1)
df_fe['mobile_wt'] = df_fe['mobile_wt'].replace(0, 0.1)

# New features
df_fe['screen_area'] = df_fe['sc_h'] * df_fe['sc_w']
df_fe['battery_per_weight'] = df_fe['battery_power'] / df_fe['mobile_wt']
df_fe['ppi'] = np.sqrt(df_fe['px_height']*2 + df_fe['px_width']*2) / df_fe['sc_w']

# Scale
from sklearn.preprocessing import StandardScaler
scale_cols = [
    'battery_power', 'ram', 'int_memory', 'mobile_wt',
    'screen_area', 'battery_per_weight', 'ppi'
]
scaler = StandardScaler()
df_fe[scale_cols] = scaler.fit_transform(df_fe[scale_cols])

# Check for any inf/NaN again (should be none)
print("Any infinities?", np.isinf(df_fe[scale_cols]).any().any())
print("Any NaNs?", df_fe[scale_cols].isna().any().any())

df_fe.head()


# In[7]:


pip install xgboost


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Split data
X = df_fe.drop('price_range', axis=1)
y = df_fe['price_range']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Store models and their names
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"🔍 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("-" * 50)


# In[9]:


X_train.head()


# In[10]:


X_test.head()


# In[11]:


from sklearn.model_selection import RandomizedSearchCV

# Define XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize RandomizedSearchCV
xgb_random = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid,
    n_iter=20, cv=3, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42
)

# Fit on training data
xgb_random.fit(X_train, y_train)

# Best model
best_xgb = xgb_random.best_estimator_

# Evaluate on test set
xgb_preds = best_xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("✅ Best XGBoost Accuracy:", round(xgb_acc, 4))
print("Classification Report:\n", classification_report(y_test, xgb_preds))


# In[12]:


import pickle

# Save the best model as a pickle file
with open("best_xgb_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

print("✅ Model saved as best_xgb_model.pkl")


# ### Set up MLflow tracking & experiment

# In[15]:


import os
print(os.getcwd())



# In[16]:


os.listdir()


# In[21]:


get_ipython().system('pip install mlflow')


import mlflow
from mlflow import sklearn as mlflow_sklearn

print(mlflow.__version__)


# In[22]:


# # (A) Local tracking store (good for SageMaker/Jupyter)
# TRACKING_DIR = "/home/ec2-user/SageMaker/mobile_prediction_folder/mlruns"  # change if needed
# os.makedirs(TRACKING_DIR, exist_ok=True)
# mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
EC2_PUBLIC_IP = "18.175.251.68"
mlflow.set_tracking_uri(f"http://{EC2_PUBLIC_IP}:5000")


# In[23]:


# (B) Name your experiment
EXPERIMENT_NAME = "xgb_random_search_experiment"
mlflow.set_experiment(EXPERIMENT_NAME)


# ### Enable autologging and run your search

# In[24]:


import mlflow.sklearn
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# your existing objects: X_train, y_train, X_test, y_test, param_grid (as you defined)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

xgb_random = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

with mlflow.start_run(run_name="xgb_random_search") as run:
    xgb_random.fit(X_train, y_train)
    best_xgb = xgb_random.best_estimator_

    # Evaluate on test set and log
    from sklearn.metrics import accuracy_score, classification_report
    xgb_preds = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    mlflow.log_metric("test_accuracy", float(xgb_acc))
    print("✅ Best XGBoost Accuracy:", round(xgb_acc, 4))
    print("Classification Report:\n", classification_report(y_test, xgb_preds))


# ### Add rich artifacts (confusion matrix, report, feature importance)

# In[25]:


import json, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

with mlflow.start_run(run_name="xgb_best_eval", nested=True) as eval_run:
    # Confusion matrix
    cm = confusion_matrix(y_test, xgb_preds)
    disp = ConfusionMatrixDisplay(cm)
    plt.figure()
    disp.plot()
    plt.title("Confusion Matrix - Best XGB")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Classification report as JSON and text
    report_dict = classification_report(y_test, xgb_preds, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    mlflow.log_artifact("classification_report.json")

    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, xgb_preds))
    mlflow.log_artifact("classification_report.txt")

    # Feature importance
    importances = getattr(best_xgb, "feature_importances_", None)
    if importances is not None:
        plt.figure()
        order = np.argsort(importances)[::-1][:30]  # top 30
        names = getattr(X_train, "columns", np.arange(len(importances)))
        names = np.array(names)[order]
        plt.bar(range(len(order)), importances[order])
        plt.xticks(range(len(order)), names, rotation=90)
        plt.title("Top Feature Importances (XGB)")
        plt.tight_layout()
        plt.savefig("feature_importances.png", bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("feature_importances.png")


# ### Log the final model with signature & input example

# In[26]:


from mlflow.models import infer_signature

# Infer signature (works whether X_* are pandas or numpy)
try:
    proba_or_pred = best_xgb.predict_proba(X_train[:5])
except Exception:
    proba_or_pred = best_xgb.predict(X_train[:5])
signature = infer_signature(X_train[:5], proba_or_pred)

with mlflow.start_run(run_name="xgb_model_logging", nested=True) as log_run:
    # Prefer xgboost flavor for XGBClassifier
    import mlflow.xgboost
    mlflow.xgboost.log_model(
        best_xgb,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:5]
    )
    RUN_ID = log_run.info.run_id
    print("Model logged under run:", RUN_ID)


# ### Register the model to the MLflow Model Registry

# In[27]:


from mlflow.tracking import MlflowClient

MODEL_NAME = "xgb_classifier_randomsearch"

client = MlflowClient()
model_uri = f"runs:/{RUN_ID}/model"

# Create the registered model if it doesn't exist
registered = [m.name for m in client.search_registered_models()]
if MODEL_NAME not in registered:
    client.create_registered_model(MODEL_NAME)

# Create a new model version
mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=RUN_ID)
print("Registered Model Version:", mv.version)

# (Optional) transition stage: "Staging" or "Production"
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="Staging",
    archive_existing_versions=False
)


# In[ ]:


# import mlflow
# #X_test = pd.read_csv("mobile_prediction_folder/test.csv") 
# # A) From a specific run (exact artifact)
# loaded = mlflow.pyfunc.load_model(model_uri=f"runs:/{RUN_ID}/model")

# # B) From registry (latest Staging version)
# loaded_from_registry = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Staging")

# # Use like a sklearn estimator
# preds = loaded_from_registry.predict(X_test)


# In[ ]:


preds


# In[ ]:




