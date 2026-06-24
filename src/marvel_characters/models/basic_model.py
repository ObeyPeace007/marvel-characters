"""Basic model implementation for Marvel character classification.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict (Alive).
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from marvel_characters.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for Marvel character survival prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.marvel_character_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.
        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.num_features + self.cat_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        class CatToIntTransformer(BaseEstimator, TransformerMixin):
            """Transformer that encodes categorical columns as integer codes for LightGBM.

            Unknown categories at transform time are encoded as -1.
            """

            def __init__(self, cat_features: list[str]) -> None:
                """Initialize the transformer with categorical feature names."""
                self.cat_features = cat_features
                self.cat_maps_ = {}

            def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
                """Fit the transformer to the DataFrame X."""
                self.fit_transform(X)
                return self

            def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
                """Fit and transform the DataFrame X."""
                X = X.copy()
                for col in self.cat_features:
                    c = pd.Categorical(X[col])
                    # Build mapping: {category: code}
                    self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                """Transform the DataFrame X by encoding categorical features as integers."""
                X = X.copy()
                for col in self.cat_features:
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

        preprocessor = ColumnTransformer(
            transformers=[("cat", CatToIntTransformer(self.cat_features), self.cat_features)], remainder="passthrough"
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")
            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )
            eval_data = self.X_test.copy()
            eval_data[self.config.target] = self.y_test

            result = mlflow.models.evaluate(
                self.model_info.model_uri,
                eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            self.metrics = result.metrics

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        latest_model_uri = f"models:/{latest_model_version.model_id}"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics
        if self.metrics["f1_score"] >= metrics_old["f1_score"]:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")


# =============================================================================
# EXPLANATION OF BASIC_MODEL.PY
# =============================================================================
#
# SECTION 1: IMPORTS
# ------------------
# mlflow              → Experiment tracking, model logging
# DeltaTable          → Access versioned data in Databricks
# LGBMClassifier      → The ML model (LightGBM)
# logger              → Pretty logging with emojis
# MlflowClient        → Programmatic model registry access
# infer_signature     → Capture input/output schema
# BaseEstimator, TransformerMixin → Create custom sklearn transformer
# ColumnTransformer, Pipeline     → Build preprocessing pipeline
#
#
# SECTION 2: __init__
# -------------------
# Unpacks config into instance variables for easy access later.
#
#   config (YAML)              BasicModel instance
#   ┌─────────────────┐        ┌─────────────────────────┐
#   │ num_features    │  ───>  │ self.num_features       │
#   │ cat_features    │  ───>  │ self.cat_features       │
#   │ target          │  ───>  │ self.target             │
#   │ parameters      │  ───>  │ self.parameters         │
#   └─────────────────┘        └─────────────────────────┘
#
#
# SECTION 3: load_data()
# ----------------------
# 1. spark.table(...)        → Load data as Spark DataFrame
# 2. .toPandas()             → Convert to pandas (sklearn needs pandas)
# 3. [features]              → Select only feature columns
# 4. DeltaTable.history()    → Get data version (tracks which data trained model)
#
# Why track data version?
# "This model was trained on version 5 of train_set" - crucial for debugging!
#
#
# SECTION 4: prepare_features()
# -----------------------------
# CatToIntTransformer: Converts categorical strings to integers for LightGBM.
#
#   Before:              After:
#   ┌──────────┐        ┌───┐
#   │ "Male"   │  ───>  │ 0 │
#   │ "Female" │  ───>  │ 1 │
#   │ "Unknown"│  ───>  │-1 │  (unknown = -1)
#   └──────────┘        └───┘
#
# Pipeline flow:
#   Input Data → ColumnTransformer → LGBMClassifier → Prediction
#   (num cols pass through, cat cols get transformed to integers)
#
#
# SECTION 5: train()
# ------------------
# Simple! Fits the entire pipeline (preprocessing + model) on training data.
#
#
# SECTION 6: log_model()
# ----------------------
# Step 1: infer_signature    → Documents what model expects as input/output
# Step 2: log_input          → Links model to exact data version (lineage)
# Step 3: log_model          → Saves pipeline + signature for later use
# Step 4: evaluate           → Auto-computes accuracy, f1, precision, recall
#
#
# SECTION 7: model_improved() - CHAMPION/CHALLENGER PATTERN
# ---------------------------------------------------------
# Compares NEW model (challenger) vs CURRENT deployed model (champion).
# Only returns True if new model has better F1 score.
#
# Flow:
#   1. Load current "latest-model" (champion) from registry
#   2. Evaluate champion on test data → get old metrics
#   3. Compare: new_f1 >= old_f1 ?
#   4. If yes → new model wins, ready to become champion
#   5. If no  → keep current champion, don't deploy new model
#
#
# SECTION 8: register_model()
# ---------------------------
# Adds model to Unity Catalog registry for deployment.
# Creates a new version (v1, v2, v3, ...) each time.
#
#
# FULL FLOW SUMMARY
# =================
#
#   BasicModel(config, tags, spark)
#          │
#          ▼
#      load_data()
#          │ Load from Databricks tables
#          ▼
#     prepare_features()
#          │ Build preprocessing pipeline
#          ▼
#        train()
#          │ Fit pipeline on data
#          ▼
#      log_model()
#          │ Log to MLflow with signature, data lineage, metrics
#          ▼
#     register_model()
#          │ Add to Unity Catalog
#          ▼
#       DONE! Model is tracked and ready for deployment
#
#
# =============================================================================
# DEEP DIVE: DATA VERSIONING WITH DELTA LAKE
# =============================================================================
#
# Delta Lake automatically versions data on every write operation.
# You don't need to manually tag data - it's built-in!
#
# HOW IT WORKS:
# -------------
# # Week 1: Initial load
# df.write.format("delta").saveAsTable("train_set")  → version 0
#
# # Week 2: Append new data
# new_data.write.mode("append").saveAsTable("train_set")  → version 1
#
# # Week 3: More updates
# more_data.write.mode("append").saveAsTable("train_set")  → version 2
#
#
# VIEW VERSION HISTORY:
# ---------------------
# delta_table = DeltaTable.forName(spark, "mlops_dev.marvel_characters.train_set")
# delta_table.history().show()
#
# Output:
# +-------+-------------------+--------+
# |version|          timestamp|operation|
# +-------+-------------------+--------+
# |      2|2026-06-24 10:00:00|   WRITE|
# |      1|2026-06-17 10:00:00|   WRITE|
# |      0|2026-06-10 10:00:00|   WRITE|
# +-------+-------------------+--------+
#
#
# TIME TRAVEL - READ SPECIFIC VERSION:
# ------------------------------------
# # Read version 1 (last week's data)
# old_data = spark.read.option("versionAsOf", 1).table("train_set")
#
# # Or by timestamp
# old_data = spark.read.option("timestampAsOf", "2026-06-17").table("train_set")
#
#
# HOW THIS FILE USES IT:
# ----------------------
# In load_data():
#   delta_table = DeltaTable.forName(spark, "train_set")
#   self.train_data_version = delta_table.history().select("version").first()[0]
#
# In log_model():
#   mlflow.log_input(dataset, context="training")
#   # Records: "Model trained on train_set @ version 2"
#
#
# WEEKLY DATA FLOW:
# -----------------
#   Week 1          Week 2          Week 3          Week 4
#   ─────────────────────────────────────────────────────────
#   Data v0         Data v1         Data v2         Data v3
#   (1000 rows)     (+300 rows)     (+500 rows)     (+200 rows)
#       │               │               │               │
#       ▼               ▼               ▼               ▼
#   Model v1        Model v2        Model v3        Model v4
#       │               │               │               │
#       └───────────────┴───────────────┴───────────────┘
#                               │
#                      MLflow tracks ALL of this!
#
#
# =============================================================================
# DEEP DIVE: CHAMPION/CHALLENGER PATTERN
# =============================================================================
#
# A production ML deployment strategy: only deploy new models if they're better.
#
# THE PROBLEM:
# ------------
# "We trained a new model. Should we deploy it?"
#
# Bad approach: Just deploy it! 😱
#   - What if it's worse than current model?
#   - Users experience degraded predictions
#   - Hard to rollback
#
# Good approach: Champion/Challenger pattern ✅
#   - Compare before deploying
#   - Only promote if proven better
#
#
# THE CONCEPT:
# ------------
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                    PRODUCTION                                   │
# │  ┌─────────────────────────────────────────────────────────┐   │
# │  │  CHAMPION (current deployed model)                       │   │
# │  │  Alias: @latest-model                                    │   │
# │  │  Version: v3                                             │   │
# │  │  F1 Score: 0.82                                          │   │
# │  │  Status: Serving 100% of traffic                         │   │
# │  └─────────────────────────────────────────────────────────┘   │
# └─────────────────────────────────────────────────────────────────┘
#                               │
#                               │ Compare
#                               ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                    STAGING / TESTING                            │
# │  ┌─────────────────────────────────────────────────────────┐   │
# │  │  CHALLENGER (newly trained model)                        │   │
# │  │  Version: v4                                             │   │
# │  │  F1 Score: 0.85  ← Better!                               │   │
# │  │  Status: Being evaluated                                 │   │
# │  └─────────────────────────────────────────────────────────┘   │
# └─────────────────────────────────────────────────────────────────┘
#
#
# THE CODE FLOW (model_improved method):
# --------------------------------------
#
# def model_improved(self) -> bool:
#     # 1. Get current CHAMPION
#     champion = client.get_model_version_by_alias(name, "latest-model")
#
#     # 2. Evaluate CHAMPION on test data
#     champion_f1 = evaluate(champion).metrics["f1_score"]  # e.g., 0.82
#
#     # 3. Compare with CHALLENGER (new model)
#     challenger_f1 = self.metrics["f1_score"]  # e.g., 0.85
#
#     # 4. Decision
#     if challenger_f1 >= champion_f1:
#         return True   # Challenger wins! Promote to champion
#     else:
#         return False  # Champion stays, discard challenger
#
#
# WEEKLY RETRAINING WORKFLOW:
# ---------------------------
#
#   Weekly Retraining Job
#            │
#            ▼
#   ┌─────────────────┐
#   │ Train new model │
#   │ (Challenger)    │
#   └────────┬────────┘
#            │
#            ▼
#   ┌─────────────────┐
#   │ model_improved()│──── No ────> Discard challenger
#   │ Compare metrics │              Keep champion
#   └────────┬────────┘
#            │ Yes
#            ▼
#   ┌─────────────────┐
#   │ register_model()│
#   │ Set alias       │
#   │ @latest-model   │
#   └────────┬────────┘
#            │
#            ▼
#      Challenger becomes
#      new Champion! 🏆
#
#
# WHY THIS MATTERS:
# -----------------
# | Without Champion/Challenger | With Champion/Challenger      |
# |-----------------------------|-------------------------------|
# | Deploy blindly              | Only deploy if proven better  |
# | Rollback is painful         | Champion always there         |
# | No comparison metrics       | Clear decision criteria       |
# | "Who broke production?"     | Automated quality gate        |
#
# =============================================================================

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
