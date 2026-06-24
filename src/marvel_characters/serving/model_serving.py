"""Model serving module for Marvel characters.

# =============================================================================
# WHAT DOES THIS FILE DO? (Simple Explanation)
# =============================================================================
#
# This file takes your trained model and puts it on the INTERNET as a URL.
#
#   BEFORE (Notebook):                    AFTER (Model Serving):
#   ────────────────────────────────      ────────────────────────────────
#   1. Open Databricks                    1. App calls URL
#   2. Open notebook                      2. Gets prediction in 50ms
#   3. Click "Run"
#   4. Wait 5 min for cluster             That's it!
#   5. See prediction
#
#   WHO USES NOTEBOOKS?                   WHO USES MODEL SERVING?
#   → Data scientists (you)               → Apps, websites, other systems
#
#
# =============================================================================
# WHAT IS AN SDK?
# =============================================================================
#
# SDK = Software Development Kit = A TOOLBOX of pre-written code
#
#   Without SDK:                          With SDK:
#   ────────────────────────────────      ────────────────────────────────
#   You have to:                          You just write:
#   • Build HTTP request manually         workspace.serving_endpoints.create(...)
#   • Handle authentication
#   • Parse JSON response                 SDK handles all the hard stuff!
#   • Handle errors
#   • ... lots of complex code
#
# Think of it like: Instead of building a car from scratch,
#                   you just use the steering wheel (SDK) to drive.
#
#
# =============================================================================
# WHY DO WE NEED MODEL SERVING?
# =============================================================================
#
#   Example: You build a Marvel Fan Website
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │   🦸 Marvel Character Survival Predictor                            │
#   │                                                                     │
#   │   Enter character details:                                          │
#   │   Appearances: [50]                                                 │
#   │   Year: [1990]                                                      │
#   │                                                                     │
#   │   [PREDICT] ← User clicks                                           │
#   │                                                                     │
#   │   Result: "This character will likely SURVIVE!"                     │
#   └─────────────────────────────────────────────────────────────────────┘
#
#   WITHOUT Model Serving:
#   → "Please wait 5 minutes while cluster starts..." → User leaves 😢
#
#   WITH Model Serving:
#   → Response in 50 milliseconds → User happy! 🎉
#
#
# =============================================================================
# CHAMPION/CHALLENGER (A/B TESTING) - WHY MODEL SERVING IS REQUIRED
# =============================================================================
#
#   100 requests come in
#        │
#        ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │                     SINGLE ENDPOINT                                 │
#   │                     (traffic router)                                │
#   │                                                                     │
#   │         90 requests              10 requests                        │
#   │              │                        │                             │
#   │              ▼                        ▼                             │
#   │   ┌─────────────────┐      ┌─────────────────┐                      │
#   │   │   Champion      │      │   Challenger    │                      │
#   │   │   (Model v1)    │      │   (Model v2)    │                      │
#   │   │   90% traffic   │      │   10% traffic   │                      │
#   │   └─────────────────┘      └─────────────────┘                      │
#   └─────────────────────────────────────────────────────────────────────┘
#
#   This lets you TEST a new model on real users before fully deploying it!
#
"""

import mlflow
from databricks.sdk import WorkspaceClient  # SDK = toolbox to talk to Databricks easily
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,  # Container for endpoint settings
    ServedEntityInput,  # Settings for ONE model being served
)


class ModelServing:
    """Manages model serving in Databricks for Marvel characters.
    
    This class does 3 simple things:
    1. Connects to Databricks
    2. Finds your model version
    3. Creates a URL so apps can call your model
    """

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        
        Example:
            serving = ModelServing(
                model_name="mlops_dev.marvel_characters.marvel_character_model",
                endpoint_name="marvel-serving-endpoint"
            )
        """
        # Connect to Databricks (SDK handles authentication automatically)
        self.workspace = WorkspaceClient()
        
        # The URL name: https://databricks.../serving-endpoints/{endpoint_name}/invocations
        self.endpoint_name = endpoint_name
        
        # Which model to serve: catalog.schema.model_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        
        HOW IT WORKS:
        ─────────────────────────────────────────────────────────────────────
        Unity Catalog has your model with multiple versions:
        
            marvel_character_model
            ├── Version 1  ◀── alias: "latest-model" (points here)
            ├── Version 2
            └── Version 3
        
        This method asks: "Which version has the 'latest-model' alias?"
        Answer: "1"
        
        WHY USE ALIAS?
        • Don't hardcode version numbers in code
        • Easy to update: just move alias to new version
        • All endpoints using "latest-model" automatically get the new version
        """
        client = mlflow.MlflowClient()
        
        # Ask MLflow: "What version has the 'latest-model' alias?"
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        
        PARAMETERS EXPLAINED:
        ─────────────────────────────────────────────────────────────────────
        
        version:
            "latest" = use the alias to find version (recommended)
            "1", "2" = use specific version number
        
        workload_size:
            "Small"  = ~4 requests/second  (dev/test, cheap)
            "Medium" = ~16 requests/second (production)
            "Large"  = ~64 requests/second (high traffic)
        
        scale_to_zero:
            True  = Shut down when no one is using it (SAVES MONEY!)
                    But: first request after idle takes ~10-30 seconds
            False = Always running (faster, but costs money 24/7)
        
        For learning: Use Small + scale_to_zero=True (very cheap!)
        """
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Check if endpoint already exists
        # ─────────────────────────────────────────────────────────────────
        # List all endpoints, check if ours is in the list
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Get the version number
        # ─────────────────────────────────────────────────────────────────
        # If "latest", look up the alias. Otherwise use the number given.
        entity_version = self.get_latest_model_version() if version == "latest" else version

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Define what model to serve and how
        # ─────────────────────────────────────────────────────────────────
        # ServedEntityInput = "Here's ONE model I want to serve"
        # You can have MULTIPLE models for A/B testing (champion/challenger)
        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,          # Which model (from Unity Catalog)
                scale_to_zero_enabled=scale_to_zero,  # Shut down when idle?
                workload_size=workload_size,          # How much compute power?
                entity_version=entity_version,        # Which version of the model?
            )
        ]

        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Create OR Update the endpoint
        # ─────────────────────────────────────────────────────────────────
        if not endpoint_exists:
            # FIRST TIME: Create new endpoint
            # After this, your model is available at:
            # POST https://databricks.../serving-endpoints/{endpoint_name}/invocations
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            # ENDPOINT EXISTS: Just update the model/version
            # Useful when you have a new model version to deploy
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)


# =============================================================================
# HOW TO USE THIS CLASS
# =============================================================================
#
# from marvel_characters.serving.model_serving import ModelServing
#
# # Create the serving manager
# serving = ModelServing(
#     model_name="mlops_dev.marvel_characters.marvel_character_model",
#     endpoint_name="marvel-serving-endpoint"
# )
#
# # Deploy the endpoint (creates URL for your model)
# serving.deploy_or_update_serving_endpoint(
#     version="latest",        # Use the "latest-model" alias
#     workload_size="Small",   # Cheap for learning
#     scale_to_zero=True       # Save money when not using
# )
#
# # Now your model is available at:
# # POST https://dbc-xxx.cloud.databricks.com/serving-endpoints/marvel-serving-endpoint/invocations
#
# # To call it:
# # response = requests.post(url, json={"dataframe_records": [{"APPEARANCES": 50, ...}]})
# # prediction = response.json()["predictions"]  # → ["alive"] or [1]
#
