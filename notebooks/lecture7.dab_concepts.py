# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Asset Bundles (DAB) - Concepts Guide
# MAGIC 
# MAGIC **Don't memorize this!** Use it as a reference when you need it.
# MAGIC 
# MAGIC The `databricks.yml` file is essentially a **template** - you can find examples online and just fill in YOUR values.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # What is DAB? (The Big Picture)
# MAGIC 
# MAGIC ## Before DAB (Manual Workflow)
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         YOUR CURRENT WORKFLOW                               │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   YOU (human)                                                               │
# MAGIC │     │                                                                       │
# MAGIC │     ├── 1. Open Databricks                                                  │
# MAGIC │     ├── 2. Open notebook                                                    │
# MAGIC │     ├── 3. Click "Run All"                                                  │
# MAGIC │     ├── 4. Wait for results                                                 │
# MAGIC │     ├── 5. Check if model registered                                        │
# MAGIC │     ├── 6. Open next notebook                                               │
# MAGIC │     ├── 7. Click "Run All"                                                  │
# MAGIC │     └── ... repeat manually ...                                             │
# MAGIC │                                                                             │
# MAGIC │   PROBLEMS:                                                                 │
# MAGIC │   ❌ You have to be there to click buttons                                  │
# MAGIC │   ❌ Can't run at 3am automatically                                         │
# MAGIC │   ❌ No version control for "how to run" things                             │
# MAGIC │   ❌ Hard to replicate: "What settings did I use last time?"                │
# MAGIC │   ❌ Dev vs Prod? You manually change configs each time                     │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## With DAB (Automated Workflow)
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         DAB + WORKFLOWS                                     │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   YOU (one time setup)                                                      │
# MAGIC │     │                                                                       │
# MAGIC │     └── Write YAML file that says:                                          │
# MAGIC │         "Run notebook A, then B, then C, every day at 3am"                  │
# MAGIC │                                                                             │
# MAGIC │   DATABRICKS (automatic, forever)                                           │
# MAGIC │     │                                                                       │
# MAGIC │     ├── 3:00 AM → Run preprocessing notebook                                │
# MAGIC │     ├── 3:15 AM → Run training notebook                                     │
# MAGIC │     ├── 3:30 AM → Run model registration                                    │
# MAGIC │     ├── 3:45 AM → Deploy to endpoint                                        │
# MAGIC │     └── Send you email: "Pipeline complete ✅"                              │
# MAGIC │                                                                             │
# MAGIC │   BENEFITS:                                                                 │
# MAGIC │   ✅ Runs automatically (no clicking)                                       │
# MAGIC │   ✅ Runs at scheduled times (3am daily, weekly, etc.)                      │
# MAGIC │   ✅ YAML is version controlled (Git tracks changes)                        │
# MAGIC │   ✅ Same YAML, different environments (dev/staging/prod)                   │
# MAGIC │   ✅ Reproducible: "Here's exactly how we run everything"                   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Simple Analogy
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                              COOKING ANALOGY                                │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   WHAT YOU'VE BEEN DOING (Manual):                                          │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   You're the chef. You cook every meal yourself.                            │
# MAGIC │   • "I'll make pasta tonight" (open notebook, run)                          │
# MAGIC │   • "Now I'll make salad" (open another notebook, run)                      │
# MAGIC │   • You have to be in the kitchen every time                                │
# MAGIC │                                                                             │
# MAGIC │   DAB/WORKFLOWS (Automated):                                                │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   You write a RECIPE BOOK (YAML) and hire a robot chef (Databricks).        │
# MAGIC │   • Recipe book says: "6pm: Make pasta. 6:30pm: Make salad."                │
# MAGIC │   • Robot follows the recipe automatically                                  │
# MAGIC │   • You can sleep while dinner is made                                      │
# MAGIC │                                                                             │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   YAML file        = The recipe book                                        │
# MAGIC │   Databricks Job   = The robot chef                                         │
# MAGIC │   DAB              = The system that gives recipes to the robot             │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Key Terms
# MAGIC 
# MAGIC | Term | What It Is |
# MAGIC |------|------------|
# MAGIC | **DAB** | Tool to package code + config + deployment instructions together |
# MAGIC | **Workflow** | A sequence of tasks (notebooks/scripts) that run automatically |
# MAGIC | **YAML** | The file format where you define everything (databricks.yml) |
# MAGIC | **Target** | Environment (dev/staging/prod) - same code, different settings |
# MAGIC | **CLI** | Command Line Interface - the `databricks` command in terminal |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Do I Need CLI?
# MAGIC 
# MAGIC **Short answer: Not always!**
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                TWO WAYS TO USE DAB/WORKFLOWS                                │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   OPTION 1: Databricks UI (no CLI needed)                                   │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   1. Go to Workflows → Create Job                                           │
# MAGIC │   2. Click "Add Task" → Select notebook                                     │
# MAGIC │   3. Click "Add Task" → Select next notebook                                │
# MAGIC │   4. Set schedule (daily at 3am)                                            │
# MAGIC │   5. Click "Create"                                                         │
# MAGIC │                                                                             │
# MAGIC │   ✅ Good for: Learning, quick setup, simple workflows                      │
# MAGIC │   ❌ Workflow config not version controlled                                 │
# MAGIC │                                                                             │
# MAGIC │   ───────────────────────────────────────────────────────────────────────   │
# MAGIC │                                                                             │
# MAGIC │   OPTION 2: YAML + CLI (or CI/CD)                                           │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   1. Write databricks.yml                                                   │
# MAGIC │   2. Run: databricks bundle deploy                                          │
# MAGIC │   (OR: GitHub Actions runs this for you automatically)                      │
# MAGIC │                                                                             │
# MAGIC │   ✅ Good for: Production, CI/CD, team collaboration                        │
# MAGIC │   ✅ Workflow config is version controlled in Git                           │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC **For CI/CD:** You don't install CLI locally - GitHub Actions runs it for you!

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # The databricks.yml File (Template)
# MAGIC 
# MAGIC Think of this as a **template you fill in**. Here's the structure:
# MAGIC 
# MAGIC ```yaml
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # SECTION 1: BUNDLE - "Who am I?"
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC bundle:
# MAGIC   name: marvel-characters              # Your project name
# MAGIC   databricks_cli_version: ">=0.218.0"  # Minimum CLI version needed
# MAGIC   git:
# MAGIC     origin_url: https://github.com/YOU/YOUR-REPO.git
# MAGIC     branch: main
# MAGIC 
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # SECTION 2: VARIABLES - "What values might change?"
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC variables:
# MAGIC   catalog_name:
# MAGIC     description: "Unity Catalog name"
# MAGIC     default: "mlops_dev"
# MAGIC   schema_name:
# MAGIC     description: "Schema name"
# MAGIC     default: "marvel_characters"
# MAGIC 
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # SECTION 3: TARGETS - "Where do I deploy? (dev vs prod)"
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC targets:
# MAGIC   dev:
# MAGIC     default: true
# MAGIC     workspace:
# MAGIC       host: https://dbc-xxx.cloud.databricks.com
# MAGIC     variables:
# MAGIC       catalog_name: "mlops_dev"
# MAGIC       
# MAGIC   prod:
# MAGIC     workspace:
# MAGIC       host: https://dbc-prod.cloud.databricks.com
# MAGIC     variables:
# MAGIC       catalog_name: "mlops_prod"
# MAGIC 
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # SECTION 4: RESOURCES - "What to run?"
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     train_model_daily:
# MAGIC       name: "Marvel Model Training"
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 3 * * ?"  # 3am daily
# MAGIC       tasks:
# MAGIC         - task_key: preprocess
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/preprocess.py
# MAGIC         - task_key: train
# MAGIC           depends_on: [preprocess]
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/train.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Bundle Section - "Who am I?"
# MAGIC 
# MAGIC ```yaml
# MAGIC bundle:
# MAGIC   name: marvel-characters              # Project name (shows in Databricks UI)
# MAGIC   databricks_cli_version: ">=0.218.0"  # Minimum CLI version required
# MAGIC   git:
# MAGIC     origin_url: https://github.com/ObeyPeace007/marvel-characters.git
# MAGIC     branch: main
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `name` | Your project's name - appears in Databricks UI |
# MAGIC | `databricks_cli_version` | Minimum CLI version needed to run this bundle |
# MAGIC | `git.origin_url` | Your GitHub repo URL |
# MAGIC | `git.branch` | Which branch to use (main, dev, feature-x) |
# MAGIC 
# MAGIC **Note:** CLI = Command Line Interface (the `databricks` command), NOT a cluster!

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Variables Section - "What values might change?"
# MAGIC 
# MAGIC ```yaml
# MAGIC variables:
# MAGIC   catalog_name:
# MAGIC     description: "The Unity Catalog name"    # Human-readable explanation
# MAGIC     default: "mlops_dev"                     # Value if not overridden
# MAGIC   
# MAGIC   schema_name:
# MAGIC     description: "The schema for tables"
# MAGIC     default: "marvel_characters"
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `description` | Human-readable explanation (for documentation) |
# MAGIC | `default` | The value to use if you don't specify one |
# MAGIC 
# MAGIC **How to use variables:**
# MAGIC ```yaml
# MAGIC notebook_task:
# MAGIC   base_parameters:
# MAGIC     catalog: ${var.catalog_name}    # ← Uses the variable!
# MAGIC ```
# MAGIC 
# MAGIC **Why variables?** Same code, different environments:
# MAGIC - Dev uses: `catalog: mlops_dev`
# MAGIC - Prod uses: `catalog: mlops_prod`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Targets Section - "Where do I deploy?"
# MAGIC 
# MAGIC ```yaml
# MAGIC targets:
# MAGIC   dev:
# MAGIC     default: true                # Use this if no target specified
# MAGIC     mode: development            # Development mode (more permissive)
# MAGIC     workspace:
# MAGIC       host: https://dev.databricks.com
# MAGIC     cluster_id: "dev-cluster-123"
# MAGIC     variables:
# MAGIC       catalog_name: "mlops_dev"
# MAGIC       
# MAGIC   prod:
# MAGIC     mode: production             # Production mode (stricter)
# MAGIC     workspace:
# MAGIC       host: https://prod.databricks.com
# MAGIC     cluster_id: "prod-cluster-456"
# MAGIC     variables:
# MAGIC       catalog_name: "mlops_prod"
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `default` | Use this target if none specified |
# MAGIC | `mode` | `development` (flexible) or `production` (strict) |
# MAGIC | `workspace.host` | Databricks workspace URL |
# MAGIC | `cluster_id` | Which cluster to use |
# MAGIC | `variables` | Override variable values for this environment |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Resources Section - "What to run?"
# MAGIC 
# MAGIC **Instructor's advice:** Stick to `jobs` and `experiments` - they cover 90% of MLOps needs.
# MAGIC 
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:           # ✅ USE THIS - Automated workflows
# MAGIC     train_model:
# MAGIC       name: "Daily Training"
# MAGIC       tasks: [...]
# MAGIC       
# MAGIC   experiments:    # ✅ USE THIS - MLflow experiments
# MAGIC     my_experiment:
# MAGIC       name: "/Shared/marvel-experiment"
# MAGIC       
# MAGIC   # These exist but AVOID for simplicity:
# MAGIC   # dashboards:   # SQL dashboards
# MAGIC   # models:       # Model registry entries  
# MAGIC   # pipelines:    # Delta Live Tables
# MAGIC   # schemas:      # Database schemas
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Permissions Section - "Who can do what?"
# MAGIC 
# MAGIC ```yaml
# MAGIC permissions:
# MAGIC   - level: CAN_VIEW           # Read-only access
# MAGIC     group_name: "data-team"   # A group of users
# MAGIC     
# MAGIC   - level: CAN_MANAGE         # Full control
# MAGIC     user_name: "enochobey@outlook.com"  # Specific user
# MAGIC     
# MAGIC   - level: CAN_RUN            # Can execute but not edit
# MAGIC     service_principal_name: "my-service-account"  # For automation/CI
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `level` | Permission type: `CAN_VIEW`, `CAN_RUN`, `CAN_MANAGE` |
# MAGIC | `group_name` | A team/group in Databricks |
# MAGIC | `user_name` | Specific person's email |
# MAGIC | `service_principal_name` | Non-human account (for CI/CD automation) |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Artifacts Section - "What to build?"
# MAGIC 
# MAGIC ```yaml
# MAGIC artifacts:
# MAGIC   my_wheel:
# MAGIC     type: whl                    # Wheel package (Python)
# MAGIC     build: python -m build       # Command to build it
# MAGIC     path: ./dist                 # Where to find the built file
# MAGIC     files:
# MAGIC       - source: ./dist/*.whl     # Which files to upload
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `type` | What kind: `whl` (Python wheel), `jar` (Java) |
# MAGIC | `build` | Command to create the artifact |
# MAGIC | `path` | Where the built files are |
# MAGIC | `files` | List of files to include |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Sync Section - "Which files to upload?"
# MAGIC 
# MAGIC ```yaml
# MAGIC sync:
# MAGIC   include:
# MAGIC     - ./notebooks/**             # Include all notebooks
# MAGIC     - ./src/**                   # Include source code
# MAGIC     
# MAGIC   exclude:
# MAGIC     - ./__pycache__/**           # Don't sync cache
# MAGIC     - ./tests/**                 # Don't sync tests
# MAGIC     - ./.git/**                  # Don't sync git folder
# MAGIC ```
# MAGIC 
# MAGIC | Field | What It Is |
# MAGIC |-------|------------|
# MAGIC | `include` | Files/folders TO sync |
# MAGIC | `exclude` | Files/folders to SKIP |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Include Section - "Split config into multiple files"
# MAGIC 
# MAGIC ```yaml
# MAGIC include:
# MAGIC   - ./resources/*.yml            # Include all YAML files from resources folder
# MAGIC   - ./configs/production.yml     # Include specific config file
# MAGIC ```
# MAGIC 
# MAGIC **Why?** For large projects, split configs into multiple files for organization.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # CLI Commands
# MAGIC 
# MAGIC ```bash
# MAGIC # 1. VALIDATE - Check YAML for errors (doesn't deploy anything)
# MAGIC databricks bundle validate
# MAGIC 
# MAGIC # 2. DEPLOY - Push everything to Databricks
# MAGIC databricks bundle deploy --target dev
# MAGIC databricks bundle deploy --target prod
# MAGIC 
# MAGIC # 3. RUN - Execute a job
# MAGIC databricks bundle run train_model --target dev
# MAGIC 
# MAGIC # 4. DESTROY - Remove everything that was deployed
# MAGIC databricks bundle destroy --target dev
# MAGIC ```
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         CLI COMMAND FLOW                                    │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   validate ──▶ deploy ──▶ run ──▶ (later) destroy                           │
# MAGIC │      │           │         │           │                                    │
# MAGIC │      │           │         │           └── Remove from Databricks           │
# MAGIC │      │           │         └── Execute the job                              │
# MAGIC │      │           └── Push code/config to Databricks                         │
# MAGIC │      └── Check YAML syntax (catches errors early)                           │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Marvel Project Pipeline Example
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         MARVEL MLOPS PIPELINE                               │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Parameters passed through pipeline:                                       │
# MAGIC │   • git_sha: Which code version                                             │
# MAGIC │   • job_run_id: Unique run identifier                                       │
# MAGIC │   • model_update: Should we update? (true/false)                            │
# MAGIC │   • model_version: Which version to deploy                                  │
# MAGIC │                                                                             │
# MAGIC │   ┌─────────────────┐    ┌─────────────────────────┐    ┌────────────────┐  │
# MAGIC │   │  PREPROCESS     │    │  TRAIN/EVAL/REGISTER    │    │    DEPLOY      │  │
# MAGIC │   │  (Lecture 2)    │───▶│  (Lecture 4)            │───▶│  (Lecture 6)   │  │
# MAGIC │   │                 │    │                         │    │                │  │
# MAGIC │   │ • Load raw data │    │ • Train new model       │    │ • Update       │  │
# MAGIC │   │ • Clean data    │    │ • Log to MLflow         │    │   endpoint     │  │
# MAGIC │   │ • Feature eng   │    │ • Compare vs champion   │    │ • New version  │  │
# MAGIC │   │ • Save to table │    │ • Register if better    │    │   goes live    │  │
# MAGIC │   └─────────────────┘    └─────────────────────────┘    └────────────────┘  │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Corresponding YAML
# MAGIC 
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     marvel_mlops_pipeline:
# MAGIC       name: "Marvel MLOps Pipeline"
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 3 * * ?"  # 3am daily
# MAGIC       
# MAGIC       tasks:
# MAGIC         - task_key: preprocess
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/lecture2.marvel_data_preprocessing.py
# MAGIC         
# MAGIC         - task_key: train_register
# MAGIC           depends_on:
# MAGIC             - task_key: preprocess   # Run AFTER preprocess completes
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/lecture4.train_register_basic_model.py
# MAGIC             base_parameters:
# MAGIC               git_sha: ${var.git_sha}
# MAGIC         
# MAGIC         - task_key: deploy
# MAGIC           depends_on:
# MAGIC             - task_key: train_register
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/lecture6.deploy_model_serving_endpoint.py
# MAGIC             base_parameters:
# MAGIC               model_version: ${tasks.train_register.outputs.model_version}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Quick Reference Summary
# MAGIC 
# MAGIC ## Sections
# MAGIC 
# MAGIC | Section | Purpose |
# MAGIC |---------|---------|
# MAGIC | `bundle` | Project name, CLI version, git info |
# MAGIC | `variables` | Reusable values that can change per environment |
# MAGIC | `targets` | Environment configs (dev/prod) |
# MAGIC | `resources` | Jobs and experiments to deploy |
# MAGIC | `permissions` | Who can view/run/manage |
# MAGIC | `artifacts` | Python wheels to build & upload |
# MAGIC | `sync` | Which files to upload to Databricks |
# MAGIC | `include` | Additional YAML files to merge |
# MAGIC 
# MAGIC ## CLI Commands
# MAGIC 
# MAGIC | Command | Purpose |
# MAGIC |---------|---------|
# MAGIC | `databricks bundle validate` | Check for errors |
# MAGIC | `databricks bundle deploy` | Push to Databricks |
# MAGIC | `databricks bundle run` | Execute a job |
# MAGIC | `databricks bundle destroy` | Remove everything |
# MAGIC 
# MAGIC ## Key Point
# MAGIC 
# MAGIC **The databricks.yml is just a template!** Find examples online, copy them, and fill in YOUR values.
# MAGIC 
# MAGIC You don't need to memorize everything - just know where to look and what values to change.
