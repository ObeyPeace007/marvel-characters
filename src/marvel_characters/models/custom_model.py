from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env  # Helper to create conda environment spec

from marvel_characters.config import Tags


def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to human-readable format.
    
    Input:  [1, 0, 1, 1]  (raw model output)
    Output: {"Survival prediction": ["alive", "dead", "alive", "alive"]}
    """
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model.
    
    PyFunc model = Wrapped model (same thing, different names!)
    
    This class wraps the basic model and transforms its output:
    - Basic model returns: [0, 1, 1, 0]
    - Wrapped model returns: {"Survival prediction": ["dead", "alive", ...]}
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the LightGBM model.
        
        Called automatically when: mlflow.pyfunc.load_model(uri)
        
        context.artifacts["lightgbm-pipeline"] - WHERE DOES THIS COME FROM?
        ─────────────────────────────────────────────────────────────────────
        YOU define it when logging the model (see log_register_model below):
            artifacts={"lightgbm-pipeline": wrapped_model_uri}
                       ↑ Your chosen name     ↑ URI to basic model
        
        At load time, MLflow provides the local path:
            context.artifacts["lightgbm-pipeline"] = "/tmp/mlflow/.../model"
        
        The key name "lightgbm-pipeline" is ARBITRARY - you could call it anything!
        """
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"])

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character.
        
        Called when: loaded_model.predict(data)
        """
        predictions = self.model.predict(model_input)  # Get raw predictions [0, 1, 1]
        return adjust_predictions(predictions)  # Convert to {"Survival...": ["dead", "alive"]}

    def log_register_model(
        self,
        wrapped_model_uri: str,  # URI of basic model to wrap, e.g., "models:/basic@latest-model"
        pyfunc_model_name: str,  # Name for new wrapped model in registry
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],  # List of .whl files to include (your custom code!)
        input_example: pd.DataFrame,
    ) -> None:
        """Log and register the model.

        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PART A: Start MLflow Run
        # Creates run named like "wrapper-lightgbm-2026-06-24" with git tags
        # ═══════════════════════════════════════════════════════════════════════
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            
            # ═══════════════════════════════════════════════════════════════════
            # PART B: Build conda environment with your wheel
            # ─────────────────────────────────────────────────────────────────
            # code_paths = ["../dist/marvel_characters-0.1.0-py3-none-any.whl"]
            #                            │
            #                            ▼ Split by "/" and get last part
            #              "marvel_characters-0.1.0-py3-none-any.whl"
            #                            │
            #                            ▼ Add "code/" prefix (MLflow convention)
            #              "code/marvel_characters-0.1.0-py3-none-any.whl"
            #
            # WHY? When deployed, MLflow needs to install your custom code.
            # The "code/" prefix tells MLflow "look in the code folder I uploaded"
            # ═══════════════════════════════════════════════════════════════════
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]  # Extract filename from path
                additional_pip_deps.append(f"code/{whl_name}")  # MLflow convention
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # ═══════════════════════════════════════════════════════════════════
            # PART C: Create signature (documents input/output schema)
            # ═══════════════════════════════════════════════════════════════════
            signature = infer_signature(
                model_input=input_example,  # Sample DataFrame
                model_output={"Survival prediction": ["alive"]}  # Sample output format
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # PART D: Log the PyFunc model
            # ─────────────────────────────────────────────────────────────────
            # What gets saved:
            #   MLflow Artifacts
            #   └── pyfunc-wrapper/
            #       ├── MLmodel                    # Metadata
            #       ├── python_model.pkl           # Serialized MarvelModelWrapper
            #       ├── conda.yaml                 # Environment spec
            #       ├── code/
            #       │   └── marvel_characters-0.1.0.whl   # Your code!
            #       └── artifacts/
            #           └── lightgbm-pipeline/     # Points to basic model
            # ═══════════════════════════════════════════════════════════════════
            model_info = mlflow.pyfunc.log_model(
                python_model=self,  # The MarvelModelWrapper instance (this class!)
                artifact_path="pyfunc-wrapper",  # Artifact folder name (NOT 'name'!)
                artifacts={"lightgbm-pipeline": wrapped_model_uri},  # KEY: defines context.artifacts in load_context!
                signature=signature,  # Input/output schema
                code_paths=code_paths,  # Your .whl file gets uploaded
                conda_env=conda_env,  # Dependencies for deployment
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # PART E: Register in Unity Catalog
        # Adds model to registry, creates version 1 (or next version)
        # ═══════════════════════════════════════════════════════════════════════
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,  # Where we just logged it
            name=pyfunc_model_name,  # e.g., "mlops_dev.marvel_characters.marvel_character_model_custom"
            tags=tags.to_dict(),
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # PART F: Set alias for easy access
        # Now loadable via: mlflow.pyfunc.load_model("models:/...@latest-model")
        # ═══════════════════════════════════════════════════════════════════════
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version



# =============================================================================
# EXPLANATION OF CUSTOM_MODEL.PY
# =============================================================================
#
# PURPOSE: Wrap the basic model to return human-readable predictions.
#
# Basic Model Output:     [0, 1, 1, 0]
# Wrapped Model Output:   {"Survival prediction": ["dead", "alive", "alive", "dead"]}
#
#
# =============================================================================
# WHAT THIS FILE DOES - VISUAL OVERVIEW
# =============================================================================
#
#   Basic Model (already trained)     Custom Wrapper              Final Output
#   ┌───────────────────────────┐    ┌───────────────────┐    ┌─────────────────────┐
#   │ Input: [features]         │    │ Takes basic model │    │ {"Survival prediction": │
#   │ Output: [0, 1, 1, 0]      │ -> │ Transforms output │ -> │  ["dead","alive",...]} │
#   │ (raw numbers)             │    │                   │    │ (human-readable)       │
#   └───────────────────────────┘    └───────────────────┘    └─────────────────────────┘
#
#
# =============================================================================
# KEY CONCEPT: WHERE DOES context.artifacts["lightgbm-pipeline"] COME FROM?
# =============================================================================
#
# YOU define it when logging the model! The name is ARBITRARY - you choose it.
#
#   LOG TIME (you define):                    LOAD TIME (MLflow provides):
#   ──────────────────────────────────────────────────────────────────────────
#   artifacts={"lightgbm-pipeline": uri}  →   context.artifacts["lightgbm-pipeline"] = "/tmp/path"
#            ↑                                           ↑
#       Your chosen name                        Same name, now with local path
#
# You could call it anything:
#   # At log time
#   artifacts={"my_cool_model": wrapped_model_uri}
#
#   # At load time (in load_context)
#   self.model = mlflow.sklearn.load_model(context.artifacts["my_cool_model"])
#
#
# =============================================================================
# KEY CONCEPT: PYFUNC MODEL vs WRAPPED MODEL - WHAT'S THE DIFFERENCE?
# =============================================================================
#
# They're the SAME THING! Different names for the same concept:
#
#   | Term           | Meaning                                           |
#   |----------------|---------------------------------------------------|
#   | PyFunc model   | MLflow's generic model format (PythonModel class) |
#   | Wrapped model  | A PyFunc model that wraps another model inside    |
#
#   MarvelModelWrapper = PyFunc model = Wrapped model (all the same!)
#            │
#            └── Contains: Basic LightGBM model (the model INSIDE)
#
# Visual:
#   ┌─────────────────────────────────────────────────┐
#   │  PyFunc Model (MarvelModelWrapper)              │
#   │  ┌───────────────────────────────────────────┐  │
#   │  │  Wrapped Model (Basic LightGBM Pipeline)  │  │ ← The model INSIDE
#   │  └───────────────────────────────────────────┘  │
#   │                                                 │
#   │  + Custom predict() logic                       │ ← Your added logic
#   │  + Human-readable output                        │
#   └─────────────────────────────────────────────────┘
#
#
# =============================================================================
# WHY code_paths IS IMPORTANT
# =============================================================================
#
# When the wrapped model is deployed (e.g., to a serving endpoint), MLflow needs
# to recreate the MarvelModelWrapper class. But MarvelModelWrapper is YOUR code,
# not part of MLflow or sklearn!
#
# Solution: Include your code as a .whl file:
#   code_paths=["../dist/marvel_characters-0.1.0-py3-none-any.whl"]
#
# What happens at deployment:
#   1. MLflow unpacks the model
#   2. Installs the .whl file: pip install marvel_characters-0.1.0.whl
#   3. Now it can import: from marvel_characters.models.custom_model import MarvelModelWrapper
#   4. Model works!
#
# Without code_paths:
#   ModuleNotFoundError: No module named 'marvel_characters'
#
#
# =============================================================================
# log_register_model() COMPLETE VISUAL FLOW
# =============================================================================
#
#   log_register_model() called
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART A: Start MLflow run                                        │
#   │   run_name = "wrapper-lightgbm-2026-06-24"                      │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART B: Build conda_env                                         │
#   │   code_paths → extract .whl name → add "code/" prefix           │
#   │   "../dist/marvel_characters-0.1.0.whl"                         │
#   │        → "marvel_characters-0.1.0.whl"                          │
#   │        → "code/marvel_characters-0.1.0.whl"                     │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART C: infer_signature                                         │
#   │   Documents: Input = DataFrame, Output = {"Survival...": [...]} │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART D: mlflow.pyfunc.log_model                                 │
#   │   Saves:                                                        │
#   │   - MarvelModelWrapper (python_model=self)                      │
#   │   - Reference to basic model (artifacts={"lightgbm-pipeline"})  │
#   │   - Your .whl file (code_paths)                                 │
#   │   - Environment (conda_env)                                     │
#   │                                                                 │
#   │   What gets saved:                                              │
#   │   MLflow Artifacts                                              │
#   │   └── pyfunc-wrapper/                                           │
#   │       ├── MLmodel                 # Metadata                    │
#   │       ├── python_model.pkl        # Serialized MarvelModelWrapper│
#   │       ├── conda.yaml              # Environment spec            │
#   │       ├── code/                                                 │
#   │       │   └── marvel_characters-0.1.0.whl  # Your code!         │
#   │       └── artifacts/                                            │
#   │           └── lightgbm-pipeline/  # Points to basic model       │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART E: mlflow.register_model                                   │
#   │   Adds to Unity Catalog: mlops_dev.marvel_characters.model_custom│
#   │   Creates version: v1                                           │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ PART F: set_registered_model_alias                              │
#   │   @latest-model → v1                                            │
#   │   Now accessible via: models:/...@latest-model                  │
#   └─────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#          DONE!
#
#
# =============================================================================
# TWO MODELS IN REGISTRY - WHY?
# =============================================================================
#
# Unity Catalog Registry
# ├── mlops_dev.marvel_characters.marvel_character_model_basic  (Basic Model)
# │   └── @latest-model → v1
# │       └── Returns: [0, 1, 1, 0]  (raw numbers)
# │
# └── mlops_dev.marvel_characters.marvel_character_model_custom  (Wrapped Model)
#     └── @latest-model → v1
#         └── References: basic model
#         └── Returns: {"Survival prediction": ["dead", "alive", ...]}
#
#
# USE CASES:
# ----------
# | Use Case                  | Which Model?   | Why?                        |
# |---------------------------|----------------|------------------------------|
# | Internal ML pipelines     | Basic model    | Faster, simpler, numeric     |
# | APIs / User-facing apps   | Wrapped model  | Human-readable output        |
# | Model serving endpoint    | Wrapped model  | Better UX for consumers      |
# | Downstream ML processing  | Basic model    | Numbers are easier to process|
#
#
# =============================================================================
# TWO WAYS TO PREDICT WITH WRAPPED MODEL
# =============================================================================
#
# After loading:
#   loaded_pyfunc = mlflow.pyfunc.load_model("models:/custom@latest-model")
#
# Way 1: Standard MLflow way (recommended)
#   result = loaded_pyfunc.predict(X_test)
#   # Returns: {"Survival prediction": ["alive", "dead", ...]}
#
# Way 2: Unwrap and call directly
#   unwrapped = loaded_pyfunc.unwrap_python_model()
#   result = unwrapped.predict(context=None, model_input=X_test)
#   # Returns: {"Survival prediction": ["alive", "dead", ...]}
#
# Both return the same result. Way 1 is cleaner for production use.
#
#
# =============================================================================
# NOTEBOOK FLOW (lecture4.train_register_custom_model.py)
# =============================================================================
#
#   Cell 1: Imports
#       │
#       ▼
#   Cell 2: Config + get wheel version
#       │   code_paths = ["../dist/marvel_characters-0.1.0.whl"]
#       ▼
#   Cell 3: Get basic model from registry
#       │   client.get_model_version_by_alias("basic", "latest-model")
#       ▼
#   Cell 4: Load test data for input_example
#       │
#       ▼
#   Cell 5: Create wrapper & log/register
#       │   wrapper = MarvelModelWrapper()
#       │   wrapper.log_register_model(...)
#       ▼
#   Cell 6: Load wrapped model
#       │   mlflow.pyfunc.load_model("models:/custom@latest-model")
#       ▼
#   Cell 7-8: Test predictions
#       │   unwrapped.predict(...)
#       │   loaded_model.predict(...)
#       ▼
#   DONE! Wrapped model ready for deployment
#
# =============================================================================
