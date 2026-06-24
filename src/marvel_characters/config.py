"""Configuration file for the Marvel characters project."""

from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.
    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    experiment_name_basic: str | None
    experiment_name_custom: str | None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]

            return cls(**config_dict)


class Tags(BaseModel):
    """Model for MLflow tags."""

    git_sha: str
    branch: str
    run_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the Tags instance to a dictionary."""
        tags_dict = {}
        tags_dict["git_sha"] = self.git_sha
        tags_dict["branch"] = self.branch
        if self.run_id is not None:
            tags_dict["run_id"] = self.run_id
        return tags_dict

# =============================================================================
# EXPLANATION OF THIS MODULE
# =============================================================================
#
# 1. @classmethod and from_yaml
# -----------------------------
# @classmethod is a method that belongs to the CLASS, not an instance.
# It receives 'cls' (the class) instead of 'self' (an instance).
# This lets you create objects in alternative ways (factory pattern).
#
# Normal way:
#   config = ProjectConfig(num_features=[...], cat_features=[...], ...)
#
# With from_yaml (easier!):
#   config = ProjectConfig.from_yaml("../project_config_marvel.yml", env="dev")
#
#
# 2. yaml.safe_load(f) - Standard PyYAML method
# ----------------------------------------------
# Reads a YAML file and converts it to a Python dictionary.
#
# YAML file:                         Python dict:
#   dev:                              {
#     catalog_name: mlops_dev           "dev": {"catalog_name": "mlops_dev"},
#   num_features:                       "num_features": ["Height", "Weight"]
#     - Height                        }
#     - Weight
#
#
# 3. config_path - Just a file path string
# -----------------------------------------
# config_path = "../project_config_marvel.yml"
# It's the relative path from the notebook to the YAML config file.
#
#
# 4. Tags Class - MLflow Metadata
# --------------------------------
# Tracks which code version produced a model for debugging and auditing.
#   - git_sha: Which commit created this model
#   - branch: Which Git branch
#   - run_id: Optional MLflow run ID
#
# Usage:
#   tags = Tags(git_sha="abcd12345", branch="main")
#   tags.to_dict()  # {"git_sha": "abcd12345", "branch": "main"}
#
# Search models by tag:
#   mlflow.search_runs(filter_string="tags.git_sha='abcd12345'")
#
#
# VISUAL SUMMARY
# ==============
#
# project_config_marvel.yml    ProjectConfig.from_yaml()     ProjectConfig object
# ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
# │ dev:                │      │ 1. Open file        │      │ num_features: [...] │
# │   catalog: mlops_dev│ ───> │ 2. yaml.safe_load() │ ───> │ cat_features: [...] │
# │ num_features:       │      │ 3. cls(**dict)      │      │ catalog_name: "..." │
# │   - Height          │      └─────────────────────┘      └─────────────────────┘
# └─────────────────────┘
#
# =============================================================================