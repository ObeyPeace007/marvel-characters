# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
import json
import os

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# from dotenv import load_dotenv

# COMMAND ----------

# Set up Databricks or local MLflow tracking
def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

mlflow.get_tracking_uri()

# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

mlflow.get_tracking_uri()

# COMMAND ----------

experiment = mlflow.set_experiment(experiment_name="/Shared/marvel-demo")
mlflow.set_experiment_tags({"repository_name": "marvelousmlops/marvel-characters"})
print(experiment)

# COMMAND ----------

# dump class attributes in a json file for visualization
os.makedirs("../demo_artifacts", exist_ok=True)
with open("../demo_artifacts/mlflow_experiment.json", "w") as json_file:
    json.dump(experiment.__dict__, json_file, indent=4)

# COMMAND ----------

# get experiment by id
mlflow.get_experiment(experiment.experiment_id)

# COMMAND ----------

# search for experiment
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='marvelousmlops/marvel-characters'"
)
print(experiments)

# COMMAND ----------

# start a run
mlflow.start_run()

# COMMAND ----------

# get active run
print(mlflow.active_run().__dict__)

# COMMAND ----------

mlflow.end_run()
print(mlflow.active_run() is None)

# COMMAND ----------

# start a run
with mlflow.start_run(
    run_name="marvel-demo-run",
    tags={"git_sha": "1234567890abcd"},
    description="marvel character prediction demo run",
) as run:
    run_id = run.info.run_id
    mlflow.log_params({"type": "marvel_demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# COMMAND ----------

print(mlflow.active_run() is None)

# COMMAND ----------

run_info = mlflow.get_run(run_id=run_id).to_dictionary()
print(run_info)

# COMMAND ----------

with open("../demo_artifacts/run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------

print(run_info["data"]["metrics"])

# COMMAND ----------

print(run_info["data"]["params"])

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/marvel-demo"],
    filter_string="tags.git_sha='1234567890abcd'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------

mlflow.start_run(run_id=run_id)

# COMMAND ----------

# this will fail: not allowed to overwrite value
#mlflow.log_param("type", "marvel_demo2")

# COMMAND ----------

mlflow.log_param(key="purpose", value="get_certified")
mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,test 1
experiment = mlflow.set_experiment(experiment_name="Model Tuning")
mlflow.set_experiment_tags({"project_name": "volupia_road_hubs"})

with mlflow.start_run(run_name="grid_search", description="tuning params for road hubs"):
    mlflow.log_params({
        "alpha": 3,
        "gamma": 0.1
    })
    ## training the model and getting the accuracy value
    mlflow.log_metrics({"accuracy":0.90, "loss": 0.1})

# COMMAND ----------

# start another run and log other things
mlflow.start_run(run_name="marvel-demo-run-extra",
                 tags={"git_sha": "1234567890abcd"},
                       description="marvel demo run with extra artifacts",)
mlflow.log_metric(key="metric3", value=3.0)
# dynamically log metric (trainings epochs)
for i in range(0,3):
    mlflow.log_metric(key="metric1", value=3.0+i/2, step=i)
mlflow.log_artifact("../demo_artifacts/mlflow_meme.jpeg")
mlflow.log_text("hello, MLflow!", "hello.txt")
mlflow.log_dict({"k": "v"}, "dict_example.json")
mlflow.log_artifacts("../demo_artifacts", artifact_path="demo_artifacts")

# COMMAND ----------

# log figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1], [2, 3])

mlflow.log_figure(fig, "figure.png")

# log image dynamically

# COMMAND ----------

import numpy as np

for i in range(0,3):
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    # mlflow.log_image does not exist; use mlflow.log_figure or mlflow.log_artifact for images
    # Example fix: use matplotlib to save and log the image
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"demo_image_{i}.png", bbox_inches='tight', pad_inches=0)
    mlflow.log_artifact(f"demo_image_{i}.png", artifact_path="images")

mlflow.end_run()

# COMMAND ----------

# other ways
from time import time

time_hour_ago = int(time() - 3600) * 1000

runs = C(
    search_all_experiments=True, #or experiment_ids=[], or experiment_names=[]
    order_by=["start_time DESC"],
    filter_string="status='FINISHED' AND "
                  f"start_time>{time_hour_ago} AND "
                  "run_name LIKE '%marvel-demo-run%' AND "
                  "metrics.metric3>0 AND "
                  "tags.mlflow.source.type!='JOB'"
)

# COMMAND ----------

runs

# COMMAND ----------

# load objects
artifact_uri = runs.artifact_uri[0]
mlflow.artifacts.load_dict(f"{artifact_uri}/dict_example.json")
# nested runs

# COMMAND ----------

mlflow.artifacts.load_image(f"{artifact_uri}/figure.png")

# COMMAND ----------

# download artifacts
mlflow.artifacts.download_artifacts(
    artifact_uri=f"{artifact_uri}/demo_artifacts",
    dst_path="../downloaded_artifacts")

# COMMAND ----------

# nested runs: useful for hyperparameter tuning
with mlflow.start_run(run_name="marvel_top_level_run") as run:
    for i in range(1,5):
        with mlflow.start_run(run_name=f"marvel_subrun_{str(i)}", nested=True) as subrun:
            mlflow.log_metrics({"m1": 5.1+i,
                                "m2": 2*i,
                                "m3": 3+1.5*i})

# COMMAND ----------

experiment = mlflow.set_experiment(experiment_name="Model Tuning")
mlflow.set_experiment_tags({"project_name": "volupia_road_hubs"})

with mlflow.start_run(run_name="", description="") as parent_run:
    for gamma in [1,2,3]:
        for alpha in [3,4,5]:
            with mlflow.start_run(run_name=f"gamma_{gamma}_alpha_{alpha}", nested=True):
                mlflow.log_params({"lr": 0.05, "max_depth":10, "gamma": gamma, "alpha": alpha})
                # training the model
                mlflow.log_metrics({"accuracy":0.90, "loss": 0.1	})

# COMMAND ----------

experiment = mlflow.set_experiment(experiment_name="/Users/enochobey@outlook.com/iris_classification")
mlflow.set_experiment_tags({})

with mlflow.start_run(run_name="epoch_test", tags={"model_type":"random_forest"}):
    mlflow.log_params({"n_estimators": 100, "max_depth":5})
    # model training
    mlflow.log_metrics({"accuracy":0.92, "f1_score": 0.89})
    mlflow.log_text("experiment logging for random forest", "experiment_log.txt")
    mlflow.log_dict({"sepal_length": 0.4, "sepal_width": 0.1, "petal_length": 0.3, "petal_width": 0.2}, "feature_importance.json")

    epochs = 10
    for epoch in range(epochs):
        loss = 1.0 - (epoch*0.08)
        accuracy = 0.5 + (epoch * 0.05)
        mlflow.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)

# COMMAND ----------

# Exercise 4: Model Comparison (Medium)
# Create 3 separate runs comparing different algorithms on the same dataset:
# Run Name	Model	Parameters	Accuracy
# run_rf	Random Forest	n_estimators=100, max_depth=10	0.89
# run_xgb	XGBoost	learning_rate=0.1, n_estimators=200	0.92
# run_lgbm	LightGBM	learning_rate=0.05, num_leaves=31	0.91
# Then use mlflow.search_runs() to find the best model by accuracy.

# COMMAND ----------

# Exercise 4: Model Comparison (Medium)
experiment = mlflow.set_experiment(experiment_name="/Users/enochobey@outlook.com/Model Comparison")
with mlflow.start_run(run_name="run_rf", tags={"model_type":"random_forest"}):
    mlflow.log_params({"n_estimators": 100, "max_depth":10})
    # model training
    mlflow.log_metrics({"accuracy":0.89})

with mlflow.start_run(run_name="run_xgb", tags={"model_type":"XGBoost"}):
    mlflow.log_params({"learning_rate": 0.1, "n_estimators": 200})
    # model training
    mlflow.log_metrics({"accuracy":0.92})

with mlflow.start_run(run_name="run_lgbm", tags={"model_type":"LightGBM"}):
    mlflow.log_params({"learning_rate": 0.05, "num_leaves":31})
    # model training
    mlflow.log_metrics({"accuracy":0.91})


# COMMAND ----------

# Exercise 5: Nested Hyperparameter Grid Search (Medium-Hard)
# Perform a grid search over:

# learning_rate: [0.01, 0.05, 0.1]
# max_depth: [3, 5, 7]
# That's 9 combinations. Use nested runs:

# Parent run: grid_search_experiment
# Child runs: One for each combination
# Log fake accuracy (use a formula like accuracy = 0.7 + learning_rate * 0.5 - max_depth * 0.02).

# After all runs complete, find the best combination using mlflow.search_runs().

# COMMAND ----------

experiment = mlflow.set_experiment(experiment_name="/Users/enochobey@outlook.com/Model_Grid_Search")
mlflow.end_run()
with mlflow.start_run(run_name="Nested Hyperparameter Grid Search", tags={"model_type":"random_forest"}) as parent_run:
    for learning_rate in [0.01, 0.05, 0.1]:
        for max_depth in [3, 5]:
            with mlflow.start_run(run_name=f"lr_{learning_rate}_depth_{max_depth}", nested=True):
                mlflow.log_params({"learning_rate": learning_rate, "max_depth": max_depth})
                # model training
                accuracy = 0.7 + learning_rate * 0.5 - max_depth * 0.02
                mlflow.log_metrics({"accuracy":accuracy})

runs_df = mlflow.search_runs(
    experiment_names=["/Users/enochobey@outlook.com/Model_Grid_Search"],
    order_by=["metrics.accuracy DESC"]
)[["run_id", "params.learning_rate", "params.max_depth", "metrics.accuracy"]]
display(runs_df)

best_run = mlflow.search_runs(
    experiment_names=["/Users/enochobey@outlook.com/Model_Grid_Search"],
    order_by=["metrics.accuracy DESC"]
).iloc[0]

print(best_run["run_id"])
print(best_run["params.learning_rate"])
print(best_run["metrics.accuracy"])

# COMMAND ----------

# Exercise 6: Complete ML Pipeline Simulation (Hard)
# Simulate a full ML workflow in one experiment:

# Data preprocessing run: Log params like test_size=0.2, scaling=standard, and metrics like n_samples=1000, n_features=10

# Feature selection run: Log which features were selected (as an artifact/dict) and the number of features kept

# Model training run (nested with tuning):

# Parent: model_training
# Children: 5 different hyperparameter combinations
# Log training time, accuracy, precision, recall for each
# Best model run:

# Log the best hyperparameters
# Log a confusion matrix as an artifact (just a dict like {"TP": 45, "FP": 5, "TN": 40, "FN": 10})
# Log a matplotlib figure showing a simple chart


# COMMAND ----------

# Exercise 7: Search and Analysis (Hard)
# After doing Exercises 4-6, practice querying:

# # Find all runs from the last hour with accuracy > 0.85
# # Find the run with the highest f1_score
# # Find all runs where learning_rate < 0.1 AND max_depth > 3
# # List all runs ordered by accuracy descending

# Hint: Use mlflow.search_runs() with filter_string and order_by

# COMMAND ----------

# Bonus Challenge: Real Model (Advanced)
# If you want to try with actual ML:

# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# import mlflow

# # Load data
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Your task: 
# 1. Create an experiment
# 2. Train the model with different hyperparameters
# 3. Log everything properly
# 4. Use mlflow.sklearn.log_model() to log the actual model
# 5. Find the best run and load the model back

# COMMAND ----------

from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
mlflow.set_registry_uri("databricks-uc")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

experiment = mlflow.set_experiment(experiment_name="/Users/enochobey@outlook.com/Iris_classifier")
mlflow.set_experiment_tags({"project_name": "Iris"})

with mlflow.start_run(run_name="data_preprocessing"):
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map(lambda x: iris.target_names[x])

    # Display dataset info
    print(f"Shape: {df.shape}")
    print(f"Classes: {list(iris.target_names)}\n")
    mlflow.log_params({"data_shape":df.shape})
    mlflow.log_dict(df['target_name'].value_counts().to_dict(), "class_counts.json")

    X = df.drop(['target', 'target_name'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.log_params({"X_train":X_train.shape[0], "X_test":X_test.shape[0], "y_train":y_train.shape[0], "y_test":y_test.shape[0], "random_state":42})
    # mlflow.log_artifact("X_train.csv", artifact_path="data")
    # mlflow.log_artifact("X_test.csv", artifact_path="data")
    # mlflow.log_artifact("y_train.csv", artifact_path="data")
    # mlflow.log_artifact("y_test.csv", artifact_path="data")


with mlflow.start_run(run_name="model_training_sweep") as parent_run:
    n_estimators = [50, 100, 150]
    max_depth = [3, 5, None]
    best_accuracy = 0
    best_params = {}
    fixed_params = {
    "random_state": 42,
    "criterion": "gini",
    "min_samples_split": 2,
    "min_samples_leaf": 1
    }

    for n in n_estimators:
        for d in max_depth:
            with mlflow.start_run(run_name=f"model_training_{n}_{d}", nested=True) as child_run:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, criterion='gini', min_samples_split=2, min_samples_leaf=1)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                acc = metrics.accuracy_score(y_test, y_pred)
                f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                precision = metrics.precision_score(y_test, y_pred, average='weighted')
                recall = metrics.recall_score(y_test, y_pred, average='weighted')
                print(f"n={n}, depth={d} | acc={acc:.4f}, f1={f1:.4f}, prec={precision:.4f}, rec={recall:.4f}")
                mlflow.log_params({"n_estimators":n, "max_depth":d})
                mlflow.log_metrics({"accuracy":acc, "f1_score":f1, "precision":precision, "recall":recall})
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators":n, "max_depth":d, **fixed_params}
                    mlflow.log_params(fixed_params)
                    mlflow.log_metric("best_accuracy", best_accuracy)
print(f"best_accuracy: {best_accuracy:.4f}, {best_params}")

with mlflow.start_run(run_name="best_model"):
    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mlflow.sklearn.log_model(rf, "model")
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    # print(f"n={n}, depth={d} | acc={acc:.4f}, f1={f1:.4f}, prec={precision:.4f}, rec={recall:.4f}")
    # mlflow.log_params({"n_estimators":n, "max_depth":d})
    # mlflow.log_metrics({"accuracy":acc, "f1_score":f1, "precision":precision, "recall":recall})
    # confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(cm)
    print(cr)
    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Iris Classification")
    plt.tight_layout()
    mlflow.log_text(cr, "classification_report.txt")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)
    print("Confusion matrix logged to MLflow using log_figure().")

# COMMAND ----------

# Phase 5: Use search_runs() to Query Your Experiments
# Try these tasks:
# Task 1: Find all runs from your experiment, ordered by accuracy (highest first)
# Task 2: Get just the top run (best accuracy) and print its run name and params
# Hint: The result is a DataFrame - use .iloc[0] to get the first row
# Task 3: Filter to only runs where accuracy > 0.9
# Hint: Use filter_string="metrics.accuracy > 0.9"
# Task 4 (Bonus): Load the model from your "best_model" run and make a prediction

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=["/Users/enochobey@outlook.com/Iris_classifier"], order_by=["metrics.accuracy DESC"])[
    ["run_id", "params.n_estimators", "params.max_depth", "metrics.accuracy"]
]
print(runs)

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=["/Users/enochobey@outlook.com/Iris_classifier"],
    order_by=["metrics.accuracy DESC"],
    filter_string="metrics.accuracy > 0.95"
)[["run_id", "params.n_estimators", "params.max_depth", "metrics.accuracy"]]
display(runs)

# COMMAND ----------

best_run = mlflow.search_runs(
    experiment_names=["/Users/enochobey@outlook.com/Iris_classifier"],
    order_by=["metrics.accuracy DESC"]
).iloc[0]
print(best_run.artifact_uri)

# COMMAND ----------

# loading the model and using it to make predictions
logged_model = mlflow.search_runs(experiment_names=["/Users/enochobey@outlook.com/Iris_classifier"], filter_string="run_name='best_model'"
).iloc[0]["artifact_uri"] + "/model"
loaded_model = mlflow.sklearn.load_model(logged_model)
loaded_model.predict(X_test)

# COMMAND ----------

# loading the model and using it to make predictions
logged_model = mlflow.search_runs(experiment_names=["/Users/enochobey@outlook.com/Iris_classifier"], filter_string="run_name='best_model'"
).iloc[0]
run_id = logged_model["run_id"]
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
loaded_model.predict(X_test)

# COMMAND ----------

predictions = loaded_model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")
print(f"Accuracy on test set: {(predictions == y_test).mean():.4f}")
