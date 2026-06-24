# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving Concepts: A Practical Guide
# MAGIC 
# MAGIC This notebook explains the **3 main patterns** for serving ML predictions in production.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## The Restaurant Analogy 🍽️
# MAGIC 
# MAGIC Think of ML serving like running a restaurant:
# MAGIC 
# MAGIC | Pattern | Restaurant Analogy | ML Equivalent |
# MAGIC |---------|-------------------|---------------|
# MAGIC | **Feature Serving** | Pre-cooked meals in the fridge | Pre-computed predictions stored in a table |
# MAGIC | **Model Serving** | Cook to order with ingredients you bring | Real-time prediction with features you provide |
# MAGIC | **Model Serving + Feature Lookup** | Cook to order, we have your usual ingredients on file | Real-time prediction, features fetched automatically |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Pattern 1: Feature Serving (Batch Predictions)
# MAGIC 
# MAGIC ## How It Works
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         FEATURE SERVING (BATCH)                             │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   OFFLINE (Nightly/Weekly Job)                                              │
# MAGIC │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
# MAGIC │   │ Feature      │    │   Model      │    │  Predictions Table           │  │
# MAGIC │   │ Table        │───▶│  (batch)     │───▶│  ┌────────┬───────────────┐  │  │
# MAGIC │   │ (all users)  │    │              │    │  │user_id │ prediction    │  │  │
# MAGIC │   └──────────────┘    └──────────────┘    │  ├────────┼───────────────┤  │  │
# MAGIC │                                           │  │ 001    │ will_churn    │  │  │
# MAGIC │                                           │  │ 002    │ wont_churn    │  │  │
# MAGIC │                                           │  │ 003    │ will_churn    │  │  │
# MAGIC │                                           │  └────────┴───────────────┘  │  │
# MAGIC │                                           └──────────────────────────────┘  │
# MAGIC │                                                         │                   │
# MAGIC │   ONLINE (Real-time Request)                            ▼                   │
# MAGIC │   ┌──────────────┐                        ┌──────────────────────────────┐  │
# MAGIC │   │ App Request  │                        │  Simple Lookup               │  │
# MAGIC │   │ "user 002"   │───────────────────────▶│  SELECT prediction           │  │
# MAGIC │   └──────────────┘                        │  WHERE user_id = '002'       │  │
# MAGIC │         │                                 └──────────────────────────────┘  │
# MAGIC │         │                                               │                   │
# MAGIC │         ▼                                               ▼                   │
# MAGIC │   ┌──────────────┐                        ┌──────────────────────────────┐  │
# MAGIC │   │ Response:    │◀───────────────────────│  Return: "wont_churn"        │  │
# MAGIC │   │ "wont_churn" │                        └──────────────────────────────┘  │
# MAGIC │   └──────────────┘                                                          │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## When to Use
# MAGIC - ✅ You know WHO you'll predict for (existing customers in your database)
# MAGIC - ✅ Predictions don't need to be real-time fresh (daily/weekly refresh is OK)
# MAGIC - ✅ High volume of lookups (millions of requests)
# MAGIC - ✅ Simple infrastructure needed
# MAGIC 
# MAGIC ## Example Use Cases
# MAGIC - "Which of our 1M customers will churn this month?" (run nightly)
# MAGIC - Credit score updates (run weekly)
# MAGIC - Product recommendations for existing users
# MAGIC 
# MAGIC ## Pros & Cons
# MAGIC | Pros | Cons |
# MAGIC |------|------|
# MAGIC | ⚡ Fast lookup (no computation) | ❌ Can't handle NEW entities |
# MAGIC | 💰 Cheap (batch compute) | ❌ Predictions may be stale |
# MAGIC | 🔧 Simple architecture | ❌ Storage costs for large tables |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Pattern 2: Model Serving (Real-Time Inference)
# MAGIC 
# MAGIC ## How It Works
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         MODEL SERVING (REAL-TIME)                           │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   ┌──────────────────────────────────────────────────────────────────────┐  │
# MAGIC │   │                     Model Serving Endpoint                           │  │
# MAGIC │   │                     (always running)                                 │  │
# MAGIC │   │   ┌────────────────────────────────────────────────────────────────┐ │  │
# MAGIC │   │   │                    Registered Model                            │ │  │
# MAGIC │   │   │                    (from Unity Catalog)                        │ │  │
# MAGIC │   │   └────────────────────────────────────────────────────────────────┘ │  │
# MAGIC │   └──────────────────────────────────────────────────────────────────────┘  │
# MAGIC │                              ▲           │                                  │
# MAGIC │                              │           │                                  │
# MAGIC │   ┌──────────────────────────┘           └──────────────────────────────┐   │
# MAGIC │   │ Request with ALL features                    Response               │   │
# MAGIC │   │                                                                     │   │
# MAGIC │   │  POST /serving-endpoints/my-model/invocations                       │   │
# MAGIC │   │  {                                          {                       │   │
# MAGIC │   │    "dataframe_records": [                     "predictions": [      │   │
# MAGIC │   │      {                                          "alive"             │   │
# MAGIC │   │        "APPEARANCES": 50,                     ]                     │   │
# MAGIC │   │        "Year": 1990,                        }                       │   │
# MAGIC │   │        "ALIGN": "Good",                                             │   │
# MAGIC │   │        ...all features...                                           │   │
# MAGIC │   │      }                                                              │   │
# MAGIC │   │    ]                                                                │   │
# MAGIC │   │  }                                                                  │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC │   App ◀─────────────────────────────────────────────────────────────────▶   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## When to Use
# MAGIC - ✅ You have ALL features available at request time
# MAGIC - ✅ Need predictions for NEW data (never seen before)
# MAGIC - ✅ Freshness matters (can't use stale predictions)
# MAGIC - ✅ Lower volume, higher value predictions
# MAGIC 
# MAGIC ## Example Use Cases
# MAGIC - "Will this NEW website visitor buy?" (you have their session data)
# MAGIC - Fraud detection on a new transaction
# MAGIC - Real-time pricing based on current conditions
# MAGIC - **Marvel Project**: Predict survival for any character with known features
# MAGIC 
# MAGIC ## Pros & Cons
# MAGIC | Pros | Cons |
# MAGIC |------|------|
# MAGIC | 🆕 Works for NEW entities | 📊 Must provide ALL features |
# MAGIC | 🔄 Always fresh predictions | 💰 Higher compute cost |
# MAGIC | 🎯 Low latency (<100ms) | 🔧 Endpoint always running |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Pattern 3: Model Serving + Feature Lookup
# MAGIC 
# MAGIC ## How It Works
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                   MODEL SERVING + FEATURE LOOKUP                            │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   ┌──────────────────────────────────────────────────────────────────────┐  │
# MAGIC │   │                     Model Serving Endpoint                           │  │
# MAGIC │   │   ┌────────────────────────────────────────────────────────────────┐ │  │
# MAGIC │   │   │  1. Receive ID  │  2. Fetch Features  │  3. Run Model          │ │  │
# MAGIC │   │   └────────────────────────────────────────────────────────────────┘ │  │
# MAGIC │   └──────────────────────────────────────────────────────────────────────┘  │
# MAGIC │              ▲                    │                        │                │
# MAGIC │              │                    ▼                        ▼                │
# MAGIC │   ┌──────────┴───────┐  ┌─────────────────────┐  ┌────────────────────┐    │
# MAGIC │   │ App Request      │  │ Online Feature Store│  │ Prediction         │    │
# MAGIC │   │ {                │  │ ┌─────────────────┐ │  │ Response           │    │
# MAGIC │   │   "user_id":     │  │ │user_id│features │ │  │ {                  │    │
# MAGIC │   │   "customer_123" │  │ ├───────┼─────────┤ │  │   "prediction":    │    │
# MAGIC │   │ }                │  │ │  123  │ age=35  │ │  │   "will_churn"     │    │
# MAGIC │   │                  │  │ │       │ spend=$ │ │  │ }                  │    │
# MAGIC │   │ (just the ID!)   │  │ │       │ visits= │ │  │                    │    │
# MAGIC │   └──────────────────┘  │ └─────────────────┘ │  └────────────────────┘    │
# MAGIC │                         └─────────────────────┘                             │
# MAGIC │                                                                             │
# MAGIC │   FLOW:                                                                     │
# MAGIC │   ───────────────────────────────────────────────────────────────────────   │
# MAGIC │   1. App sends ONLY the user_id (not all features)                          │
# MAGIC │   2. Endpoint looks up features from Online Feature Store                   │
# MAGIC │   3. Model runs prediction with fetched features                            │
# MAGIC │   4. Response returned to app                                               │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## When to Use
# MAGIC - ✅ You only have an ID at request time (not all features)
# MAGIC - ✅ Features are already stored in a feature store
# MAGIC - ✅ Features change frequently (store keeps them updated)
# MAGIC - ✅ Want simple API (just send ID, not 100 features)
# MAGIC 
# MAGIC ## Example Use Cases
# MAGIC - "Customer 123 just called - will they churn?" (you just have their ID)
# MAGIC - Real-time fraud check when user logs in (lookup their history)
# MAGIC - Personalized recommendations when user opens app
# MAGIC 
# MAGIC ## Pros & Cons
# MAGIC | Pros | Cons |
# MAGIC |------|------|
# MAGIC | 📝 Simple requests (just ID) | 🔧 Complex setup |
# MAGIC | 🔄 Features always fresh | 💰 Feature store costs |
# MAGIC | 🎯 Single source of truth | ❌ Only works for known entities |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Quick Decision Guide
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                        WHICH PATTERN SHOULD I USE?                          │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │                    Do you need real-time predictions?                       │
# MAGIC │                                   │                                         │
# MAGIC │                    ┌──────────────┴──────────────┐                          │
# MAGIC │                    │                             │                          │
# MAGIC │                   NO                            YES                         │
# MAGIC │                    │                             │                          │
# MAGIC │                    ▼                             ▼                          │
# MAGIC │         ┌──────────────────┐      Do you have ALL features at request?     │
# MAGIC │         │ FEATURE SERVING  │                     │                          │
# MAGIC │         │ (Batch)          │      ┌──────────────┴──────────────┐           │
# MAGIC │         │                  │      │                             │           │
# MAGIC │         │ Pre-compute      │     YES                           NO           │
# MAGIC │         │ predictions      │      │                             │           │
# MAGIC │         │ nightly/weekly   │      ▼                             ▼           │
# MAGIC │         └──────────────────┘ ┌──────────────┐  ┌──────────────────────────┐ │
# MAGIC │                              │MODEL SERVING │  │MODEL SERVING +           │ │
# MAGIC │                              │              │  │FEATURE LOOKUP            │ │
# MAGIC │                              │Send features │  │                          │ │
# MAGIC │                              │get prediction│  │Send ID, endpoint fetches │ │
# MAGIC │                              └──────────────┘  │features automatically    │ │
# MAGIC │                                                └──────────────────────────┘ │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Summary Table
# MAGIC 
# MAGIC | Scenario | Best Pattern |
# MAGIC |----------|--------------|
# MAGIC | Predict for ALL existing customers nightly | **Feature Serving** (batch) |
# MAGIC | Predict for NEW visitors with full data | **Model Serving** |
# MAGIC | Predict for known customer, only have ID | **Model Serving + Feature Lookup** |
# MAGIC | High volume, low latency lookups | **Feature Serving** |
# MAGIC | Low volume, high value predictions | **Model Serving** |
# MAGIC | Features change frequently | **Model Serving + Feature Lookup** |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Marvel Project: We Use Model Serving (Pattern 2)
# MAGIC 
# MAGIC ## Why Model Serving?
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                    MARVEL CHARACTER SURVIVAL PREDICTION                     │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Request:                           Response:                              │
# MAGIC │   ┌────────────────────────────┐    ┌────────────────────────────────────┐  │
# MAGIC │   │ {                          │    │ {                                  │  │
# MAGIC │   │   "APPEARANCES": 50,       │    │   "Survival prediction": "alive"   │  │
# MAGIC │   │   "Year": 1990,            │───▶│ }                                  │  │
# MAGIC │   │   "ALIGN": "Good",         │    │                                    │  │
# MAGIC │   │   "EYE": "Blue",           │    │ The custom model transforms:       │  │
# MAGIC │   │   "HAIR": "Brown",         │    │   [1] → {"Survival...": "alive"}   │  │
# MAGIC │   │   "SEX": "Male",           │    │   [0] → {"Survival...": "dead"}    │  │
# MAGIC │   │   "GSM": "Heterosexual",   │    │                                    │  │
# MAGIC │   │   "ALIVE": "Living"        │    └────────────────────────────────────┘  │
# MAGIC │   │ }                          │                                            │
# MAGIC │   └────────────────────────────┘                                            │
# MAGIC │                                                                             │
# MAGIC │   WHY MODEL SERVING?                                                        │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   ✅ Can predict for ANY character (new or existing)                        │
# MAGIC │   ✅ We have all features at request time                                   │
# MAGIC │   ✅ Real-time prediction needed                                            │
# MAGIC │   ✅ No feature store setup required                                        │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Architecture Overview
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         OUR MLOPS PIPELINE                                  │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   1. TRAIN & REGISTER                                                       │
# MAGIC │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │
# MAGIC │   │ Training    │    │ Basic Model │    │ Unity Catalog Model Registry    │ │
# MAGIC │   │ Data        │───▶│ (LightGBM)  │───▶│ ┌─────────────────────────────┐ │ │
# MAGIC │   │             │    │             │    │ │ marvel_character_model      │ │ │
# MAGIC │   └─────────────┘    └─────────────┘    │ │   @latest-model (v1)        │ │ │
# MAGIC │                                         │ └─────────────────────────────┘ │ │
# MAGIC │                             │           │ ┌─────────────────────────────┐ │ │
# MAGIC │                             │           │ │ marvel_character_model_     │ │ │
# MAGIC │                             └──────────▶│ │ custom @latest-model (v1)   │ │ │
# MAGIC │                         (wrap with      │ │ (PyFunc wrapper)            │ │ │
# MAGIC │                          PyFunc)        │ └─────────────────────────────┘ │ │
# MAGIC │                                         └─────────────────────────────────┘ │
# MAGIC │                                                        │                    │
# MAGIC │   2. DEPLOY TO ENDPOINT                                ▼                    │
# MAGIC │                                         ┌─────────────────────────────────┐ │
# MAGIC │                                         │ Model Serving Endpoint          │ │
# MAGIC │                                         │ "marvel-character-endpoint"     │ │
# MAGIC │                                         │                                 │ │
# MAGIC │                                         │ ┌───────────────────────────┐   │ │
# MAGIC │                                         │ │ Champion: basic (90%)     │   │ │
# MAGIC │                                         │ │ Challenger: custom (10%)  │   │ │
# MAGIC │                                         │ └───────────────────────────┘   │ │
# MAGIC │                                         └─────────────────────────────────┘ │
# MAGIC │                                                        │                    │
# MAGIC │   3. SERVE PREDICTIONS                                 ▼                    │
# MAGIC │                                         ┌─────────────────────────────────┐ │
# MAGIC │   ┌─────────────┐                       │ App / API Consumer              │ │
# MAGIC │   │ POST /...   │◀─────────────────────▶│ "Is Spider-Man alive?"         │ │
# MAGIC │   │ {features}  │                       │                                 │ │
# MAGIC │   └─────────────┘                       └─────────────────────────────────┘ │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Databricks Components for Model Serving
# MAGIC 
# MAGIC ## Key Components
# MAGIC 
# MAGIC | Component | Purpose | Location |
# MAGIC |-----------|---------|----------|
# MAGIC | **Unity Catalog** | Store registered models | `mlops_dev.marvel_characters.model_name` |
# MAGIC | **Model Serving Endpoint** | Host model for real-time inference | Databricks Serving UI |
# MAGIC | **MLflow** | Track experiments, log models | Integrated with Unity Catalog |
# MAGIC 
# MAGIC ## Endpoint Configuration Options
# MAGIC 
# MAGIC ```python
# MAGIC # Endpoint can have multiple "served models" for A/B testing:
# MAGIC served_models = [
# MAGIC     {
# MAGIC         "model_name": "marvel_character_model",       # Champion
# MAGIC         "model_version": "1",
# MAGIC         "workload_size": "Small",
# MAGIC         "scale_to_zero_enabled": True,
# MAGIC         "traffic_percentage": 90                      # 90% of traffic
# MAGIC     },
# MAGIC     {
# MAGIC         "model_name": "marvel_character_model_custom", # Challenger  
# MAGIC         "model_version": "1",
# MAGIC         "workload_size": "Small",
# MAGIC         "scale_to_zero_enabled": True,
# MAGIC         "traffic_percentage": 10                      # 10% of traffic
# MAGIC     }
# MAGIC ]
# MAGIC ```
# MAGIC 
# MAGIC ## A/B Testing (Champion vs Challenger)
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                           A/B TESTING FLOW                                  │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Incoming Request                                                          │
# MAGIC │        │                                                                    │
# MAGIC │        ▼                                                                    │
# MAGIC │   ┌─────────────────────────────────────────────────────────────────────┐   │
# MAGIC │   │                     Traffic Router                                  │   │
# MAGIC │   │                                                                     │   │
# MAGIC │   │   ┌─────────────────────────┐    ┌─────────────────────────┐        │   │
# MAGIC │   │   │                         │    │                         │        │   │
# MAGIC │   │   │   90% → Champion        │    │   10% → Challenger      │        │   │
# MAGIC │   │   │   (basic model v1)      │    │   (custom model v1)     │        │   │
# MAGIC │   │   │                         │    │                         │        │   │
# MAGIC │   │   │   Returns: [0, 1, 1]    │    │   Returns:              │        │   │
# MAGIC │   │   │                         │    │   {"Survival...":       │        │   │
# MAGIC │   │   │                         │    │    ["dead","alive"]}    │        │   │
# MAGIC │   │   │                         │    │                         │        │   │
# MAGIC │   │   └─────────────────────────┘    └─────────────────────────┘        │   │
# MAGIC │   │                                                                     │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC │   WHY A/B TEST?                                                             │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   • Test new model on small % of traffic before full rollout                │
# MAGIC │   • Compare performance metrics (latency, accuracy, errors)                 │
# MAGIC │   • Gradual rollout: 10% → 50% → 100% if successful                         │
# MAGIC │   • Easy rollback: just shift traffic back to champion                      │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Serverless Model Serving on Databricks
# MAGIC 
# MAGIC ## What is Serverless Serving?
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                    SERVERLESS vs TRADITIONAL COMPUTE                        │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   TRADITIONAL (Clusters)              SERVERLESS (Model Serving)            │
# MAGIC │   ─────────────────────────────────   ─────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   ┌─────────────────────────────┐    ┌─────────────────────────────────┐    │
# MAGIC │   │ You configure:              │    │ Databricks manages:             │    │
# MAGIC │   │ • Node type (i3.xlarge)     │    │ • Infrastructure                │    │
# MAGIC │   │ • Number of workers         │    │ • Scaling                       │    │
# MAGIC │   │ • Runtime version           │    │ • Runtime                       │    │
# MAGIC │   │ • Auto-scaling rules        │    │ • Load balancing                │    │
# MAGIC │   │ • Spot vs on-demand         │    │ • High availability             │    │
# MAGIC │   └─────────────────────────────┘    └─────────────────────────────────┘    │
# MAGIC │                                                                             │
# MAGIC │   YOU: "I want 4 i3.xlarge       YOU: "I want Small/Medium/Large"           │
# MAGIC │         nodes with DBR 14.3"          Databricks figures out the rest       │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Why Can't You Choose Runtime/Cluster?
# MAGIC 
# MAGIC **Yes, you're right** - with serverless model serving, you **cannot** choose:
# MAGIC - ❌ Specific cluster type (e.g., i3.xlarge, r5.2xlarge)
# MAGIC - ❌ Databricks Runtime version (e.g., DBR 14.3)
# MAGIC - ❌ Number of nodes
# MAGIC - ❌ Spark configuration
# MAGIC 
# MAGIC **Why?** Because model serving endpoints are:
# MAGIC - **Not Spark clusters** - they're containerized microservices
# MAGIC - **Optimized for inference** - low latency, not batch processing
# MAGIC - **Managed by Databricks** - they handle scaling, failover, etc.
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                     CLUSTERS vs SERVING ENDPOINTS                           │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   DATABRICKS CLUSTER                  MODEL SERVING ENDPOINT                │
# MAGIC │   (for training/batch)                (for real-time inference)             │
# MAGIC │   ─────────────────────────────────   ─────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   ┌─────────────────────────────┐    ┌─────────────────────────────────┐    │
# MAGIC │   │ • Full Spark environment    │    │ • Single-container runtime      │    │
# MAGIC │   │ • Multiple nodes            │    │ • No Spark overhead             │    │
# MAGIC │   │ • You choose everything     │    │ • Optimized for <100ms latency  │    │
# MAGIC │   │ • High startup time (mins)  │    │ • Fast cold start (~seconds)    │    │
# MAGIC │   │ • Good for: training,       │    │ • Good for: real-time APIs,     │    │
# MAGIC │   │   batch inference, ETL      │    │   low-latency predictions       │    │
# MAGIC │   └─────────────────────────────┘    └─────────────────────────────────┘    │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Workload Sizes: Small, Medium, Large
# MAGIC 
# MAGIC ## What You CAN Control
# MAGIC 
# MAGIC Instead of choosing cluster specs, you choose **workload size**:
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         WORKLOAD SIZE OPTIONS                               │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Size       │ Concurrency │ Use Case                   │ Approx. Cost     │
# MAGIC │   ──────────────────────────────────────────────────────────────────────    │
# MAGIC │   Small      │ ~4 QPS      │ Dev/test, low traffic      │ $                │
# MAGIC │   Medium     │ ~16 QPS     │ Production, moderate       │ $$               │
# MAGIC │   Large      │ ~64 QPS     │ High traffic apps          │ $$$              │
# MAGIC │                                                                             │
# MAGIC │   Note: QPS = Queries Per Second (approximate, depends on model complexity) │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Scale to Zero
# MAGIC 
# MAGIC ```python
# MAGIC # When enabled, endpoint shuts down when idle (saves $$$)
# MAGIC ServedEntityInput(
# MAGIC     entity_name="my_model",
# MAGIC     workload_size="Small",
# MAGIC     scale_to_zero_enabled=True   # <-- No cost when not in use!
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC **Trade-off:**
# MAGIC - ✅ **Pro**: No cost during idle time
# MAGIC - ❌ **Con**: Cold start latency (~10-30 seconds for first request after idle)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # What If I Need More Power? (GPU, Large Clusters)
# MAGIC 
# MAGIC ## Option 1: GPU Model Serving (for Deep Learning)
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         GPU MODEL SERVING                                   │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   WHEN TO USE GPU:                                                          │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   • Deep learning models (PyTorch, TensorFlow)                              │
# MAGIC │   • Large Language Models (LLMs)                                            │
# MAGIC │   • Computer vision (image classification, object detection)                │
# MAGIC │   • Models with millions/billions of parameters                             │
# MAGIC │                                                                             │
# MAGIC │   NOT NEEDED FOR:                                                           │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   • Tree-based models (LightGBM, XGBoost, Random Forest) ← OUR MODEL        │
# MAGIC │   • Simple sklearn models                                                   │
# MAGIC │   • Small neural networks                                                   │
# MAGIC │                                                                             │
# MAGIC │   GPU Workload Sizes:                                                       │
# MAGIC │   ┌─────────────────────────────────────────────────────────────────────┐   │
# MAGIC │   │ GPU_SMALL  │ 1x T4 GPU   │ Small DL models, testing               │   │
# MAGIC │   │ GPU_MEDIUM │ 1x A10G GPU │ Medium models, production              │   │
# MAGIC │   │ GPU_LARGE  │ 1x A100 GPU │ Large models, LLMs                     │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ```python
# MAGIC # GPU endpoint example (for deep learning models)
# MAGIC ServedEntityInput(
# MAGIC     entity_name="my_pytorch_model",
# MAGIC     workload_size="GPU_SMALL",      # <-- GPU workload
# MAGIC     scale_to_zero_enabled=True
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ## Option 2: Provisioned Throughput (Predictable Performance)
# MAGIC 
# MAGIC For **Foundation Models** and **high-traffic production**, use provisioned throughput:
# MAGIC 
# MAGIC ```python
# MAGIC # Provisioned throughput - guaranteed capacity
# MAGIC ServedEntityInput(
# MAGIC     entity_name="my_model",
# MAGIC     min_provisioned_throughput=100,  # Minimum guaranteed QPS
# MAGIC     max_provisioned_throughput=500,  # Maximum scale
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ## Option 3: Custom Cluster Serving (Advanced)
# MAGIC 
# MAGIC For **very specific requirements**, you can use cluster-based serving:
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │   ⚠️  CUSTOM CLUSTER SERVING (Not recommended for most use cases)          │
# MAGIC │                                                                             │
# MAGIC │   When serverless doesn't fit:                                              │
# MAGIC │   • Need specific Python packages not in serverless                         │
# MAGIC │   • Need custom system libraries                                            │
# MAGIC │   • Compliance requirements for specific infrastructure                     │
# MAGIC │                                                                             │
# MAGIC │   Trade-offs:                                                               │
# MAGIC │   ✅ Full control over environment                                          │
# MAGIC │   ❌ You manage scaling, availability, updates                              │
# MAGIC │   ❌ Higher operational overhead                                            │
# MAGIC │   ❌ Slower cold starts                                                     │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Capacity Planning: QPS, Latency, and Compute Units
# MAGIC 
# MAGIC ## Key Terms
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         SERVING METRICS EXPLAINED                           │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   QPS (Queries Per Second)                                                  │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   How many prediction requests your endpoint can handle per second.         │
# MAGIC │                                                                             │
# MAGIC │   Example: QPS = 100 means 100 predictions per second                       │
# MAGIC │            = 6,000 per minute                                               │
# MAGIC │            = 360,000 per hour                                               │
# MAGIC │            = 8.6 million per day                                            │
# MAGIC │                                                                             │
# MAGIC │   ───────────────────────────────────────────────────────────────────────   │
# MAGIC │                                                                             │
# MAGIC │   Latency (Response Time)                                                   │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   How long it takes to return a prediction (measured in milliseconds).      │
# MAGIC │                                                                             │
# MAGIC │   • P50 latency: 50% of requests are faster than this                       │
# MAGIC │   • P95 latency: 95% of requests are faster than this                       │
# MAGIC │   • P99 latency: 99% of requests are faster than this                       │
# MAGIC │                                                                             │
# MAGIC │   Example: P50=20ms, P95=50ms, P99=100ms                                    │
# MAGIC │            Most requests: 20ms, occasional slow: 100ms                      │
# MAGIC │                                                                             │
# MAGIC │   ───────────────────────────────────────────────────────────────────────   │
# MAGIC │                                                                             │
# MAGIC │   Concurrency                                                               │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   How many requests can be processed simultaneously.                        │
# MAGIC │                                                                             │
# MAGIC │   Formula: Concurrency = QPS × Latency (in seconds)                         │
# MAGIC │                                                                             │
# MAGIC │   Example: 100 QPS × 0.05s (50ms) = 5 concurrent requests                   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## The Capacity Formula
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                      CALCULATING REQUIRED CAPACITY                          │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   THE FORMULA:                                                              │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │                    Required QPS                                             │
# MAGIC │   Units Needed = ─────────────────                                          │
# MAGIC │                   QPS per Unit                                              │
# MAGIC │                                                                             │
# MAGIC │                                                                             │
# MAGIC │   WHERE:                                                                    │
# MAGIC │   • Required QPS = Your expected traffic (requests per second)              │
# MAGIC │   • QPS per Unit = Capacity of one compute unit (depends on model)          │
# MAGIC │                                                                             │
# MAGIC │   ───────────────────────────────────────────────────────────────────────   │
# MAGIC │                                                                             │
# MAGIC │   EXAMPLE CALCULATION:                                                      │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   Scenario: E-commerce site during Black Friday                             │
# MAGIC │                                                                             │
# MAGIC │   Step 1: Estimate traffic                                                  │
# MAGIC │   ┌─────────────────────────────────────────────────────────────────────┐   │
# MAGIC │   │ • 1 million page views per hour                                     │   │
# MAGIC │   │ • 10% need a prediction (product recommendations)                   │   │
# MAGIC │   │ • = 100,000 predictions per hour                                    │   │
# MAGIC │   │ • = 100,000 / 3600 = ~28 QPS                                        │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC │   Step 2: Add safety margin (2x for spikes)                                 │
# MAGIC │   ┌─────────────────────────────────────────────────────────────────────┐   │
# MAGIC │   │ • Target QPS = 28 × 2 = 56 QPS                                      │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC │   Step 3: Choose workload size                                              │
# MAGIC │   ┌─────────────────────────────────────────────────────────────────────┐   │
# MAGIC │   │ • Small (~4 QPS) → Need 14 units  ❌ Too many                       │   │
# MAGIC │   │ • Medium (~16 QPS) → Need 4 units ✅ Reasonable                     │   │
# MAGIC │   │ • Large (~64 QPS) → Need 1 unit   ✅ Best choice                    │   │
# MAGIC │   └─────────────────────────────────────────────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Real-World Sizing Guide
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         WORKLOAD SIZE GUIDE                                 │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Traffic Level          │ Workload Size │ Scale to Zero │ Notes           │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   Dev/Testing            │ Small         │ Yes           │ Cost-effective  │
# MAGIC │   <1,000 req/day         │               │               │                 │
# MAGIC │                          │               │               │                 │
# MAGIC │   Low Production         │ Small         │ Optional      │ May have cold   │
# MAGIC │   1,000-10,000 req/day   │               │               │ start issues    │
# MAGIC │                          │               │               │                 │
# MAGIC │   Medium Production      │ Medium        │ No            │ Steady traffic  │
# MAGIC │   10,000-100,000 req/day │               │               │                 │
# MAGIC │                          │               │               │                 │
# MAGIC │   High Production        │ Large         │ No            │ Always-on       │
# MAGIC │   >100,000 req/day       │               │               │                 │
# MAGIC │                          │               │               │                 │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   FOR OUR MARVEL PROJECT:                                                   │
# MAGIC │   • Learning/Demo → Small with scale_to_zero_enabled=True                   │
# MAGIC │   • Very low cost, fine for occasional testing                              │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Cost Estimation
# MAGIC 
# MAGIC ## How Databricks Model Serving is Billed
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         BILLING MODEL                                       │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   SERVERLESS MODEL SERVING:                                                 │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │   Billed by: DBU (Databricks Unit) per hour                                 │
# MAGIC │                                                                             │
# MAGIC │   ┌───────────────────────────────────────────────────────────────────┐     │
# MAGIC │   │ Workload Size │ DBU/hour (approx) │ Monthly cost (24/7)*          │     │
# MAGIC │   │───────────────────────────────────────────────────────────────────│     │
# MAGIC │   │ Small         │ ~0.07 DBU/hr      │ ~$35-50/month                 │     │
# MAGIC │   │ Medium        │ ~0.28 DBU/hr      │ ~$140-200/month               │     │
# MAGIC │   │ Large         │ ~1.12 DBU/hr      │ ~$560-800/month               │     │
# MAGIC │   │───────────────────────────────────────────────────────────────────│     │
# MAGIC │   │ GPU_SMALL     │ ~0.65 DBU/hr      │ ~$325-450/month               │     │
# MAGIC │   │ GPU_MEDIUM    │ ~1.30 DBU/hr      │ ~$650-900/month               │     │
# MAGIC │   │ GPU_LARGE     │ ~2.60 DBU/hr      │ ~$1,300-1,800/month           │     │
# MAGIC │   └───────────────────────────────────────────────────────────────────┘     │
# MAGIC │                                                                             │
# MAGIC │   *Costs vary by cloud provider and region. Check Databricks pricing.       │
# MAGIC │                                                                             │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   SCALE TO ZERO SAVINGS:                                                    │
# MAGIC │   ─────────────────────────────────────────────────────────────────────     │
# MAGIC │                                                                             │
# MAGIC │   ┌────────────────────────────────────────────────────────────────────┐    │
# MAGIC │   │                                                                    │    │
# MAGIC │   │   Without scale_to_zero (24/7 running):                            │    │
# MAGIC │   │   Small endpoint: ~$50/month                                       │    │
# MAGIC │   │                                                                    │    │
# MAGIC │   │   With scale_to_zero (8 hours/day active):                         │    │
# MAGIC │   │   Small endpoint: ~$50 × (8/24) = ~$17/month                       │    │
# MAGIC │   │                                                                    │    │
# MAGIC │   │   For learning/dev: endpoint active 1 hour/day                     │    │
# MAGIC │   │   Small endpoint: ~$50 × (1/24) = ~$2/month  ✅ Very cheap!        │    │
# MAGIC │   │                                                                    │    │
# MAGIC │   └────────────────────────────────────────────────────────────────────┘    │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Model Serving vs Loading Model Yourself
# MAGIC 
# MAGIC ## The Key Distinction
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │      mlflow.pyfunc.load_model()          vs        Model Serving            │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   LOAD MODEL YOURSELF                    MODEL SERVING ENDPOINT             │
# MAGIC │   ─────────────────────────────────      ─────────────────────────────────  │
# MAGIC │                                                                             │
# MAGIC │   Downloads model to YOUR machine        Model lives on Databricks          │
# MAGIC │   (or notebook)                          server 24/7                        │
# MAGIC │                                                                             │
# MAGIC │   YOU run the prediction                 DATABRICKS runs prediction         │
# MAGIC │   model.predict(data)                    for anyone who calls URL           │
# MAGIC │                                                                             │
# MAGIC │   Only YOU can use it                    ANY app can call the URL           │
# MAGIC │   (need Python, MLflow, credentials)     (just needs HTTP)                  │
# MAGIC │                                                                             │
# MAGIC │   Good for: Testing, notebooks           Good for: Apps, production         │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC ## Analogy
# MAGIC 
# MAGIC | Method | Analogy |
# MAGIC |--------|---------|
# MAGIC | `mlflow.pyfunc.load_model()` | **Download a recipe** and cook at home |
# MAGIC | `MlflowClient` | **Browse the cookbook** (registry), pick what you want |
# MAGIC | **Model Serving** | **Order from a restaurant** - they cook, you just eat |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code Comparison
# MAGIC 
# MAGIC ```python
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # METHOD 1: Load model yourself (notebook/local)
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC import mlflow
# MAGIC 
# MAGIC # YOU download and run the model
# MAGIC model = mlflow.pyfunc.load_model("models:/my_model@latest-model")
# MAGIC prediction = model.predict(data)  # Runs on YOUR machine
# MAGIC 
# MAGIC 
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC # METHOD 2: Model Serving (API call)
# MAGIC # ═══════════════════════════════════════════════════════════════════════════
# MAGIC import requests
# MAGIC 
# MAGIC # You just CALL a URL - Databricks runs the model for you
# MAGIC response = requests.post(
# MAGIC     "https://databricks.../serving-endpoints/my-endpoint/invocations",
# MAGIC     headers={"Authorization": f"Bearer {token}"},
# MAGIC     json={"dataframe_records": data}
# MAGIC )
# MAGIC prediction = response.json()["predictions"]  # Databricks computed this
# MAGIC ```
# MAGIC 
# MAGIC ## When to Use Which?
# MAGIC 
# MAGIC | Scenario | Use |
# MAGIC |----------|-----|
# MAGIC | Testing in notebook | `mlflow.pyfunc.load_model()` |
# MAGIC | Building a website/app | **Model Serving** (API) |
# MAGIC | Batch predictions (1M rows) | `load_model()` or Spark |
# MAGIC | Real-time predictions | **Model Serving** (API) |
# MAGIC | A/B testing (champion/challenger) | **Model Serving** |
# MAGIC | Only you need predictions | `load_model()` |
# MAGIC | Many apps/users need predictions | **Model Serving** |
# MAGIC 
# MAGIC ## Summary
# MAGIC 
# MAGIC **Model Serving** = "Make my model available as a URL so apps can use it without needing Python/MLflow"
# MAGIC 
# MAGIC - `load_model()` → YOU download model, YOU run predictions
# MAGIC - Model Serving → Databricks hosts model, ANYONE can call URL

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Next Steps
# MAGIC 
# MAGIC Now that you understand the concepts, proceed to:
# MAGIC 
# MAGIC 1. **`lecture6.deploy_model_serving_endpoint.py`** - Deploy your model to a serving endpoint
# MAGIC 2. **`lecture6.ab_testing.py`** - Set up A/B testing between champion and challenger models
# MAGIC 
# MAGIC ## Code Preview: Creating an Endpoint
# MAGIC 
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# MAGIC 
# MAGIC # Initialize client
# MAGIC w = WorkspaceClient()
# MAGIC 
# MAGIC # Create endpoint with model
# MAGIC w.serving_endpoints.create(
# MAGIC     name="marvel-character-endpoint",
# MAGIC     config=EndpointCoreConfigInput(
# MAGIC         served_entities=[
# MAGIC             ServedEntityInput(
# MAGIC                 entity_name="mlops_dev.marvel_characters.marvel_character_model",
# MAGIC                 entity_version="1",
# MAGIC                 workload_size="Small",
# MAGIC                 scale_to_zero_enabled=True
# MAGIC             )
# MAGIC         ]
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ## Code Preview: Querying an Endpoint
# MAGIC 
# MAGIC ```python
# MAGIC import requests
# MAGIC 
# MAGIC # Get token and host
# MAGIC token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# MAGIC host = spark.conf.get("spark.databricks.workspaceUrl")
# MAGIC 
# MAGIC # Make prediction request
# MAGIC response = requests.post(
# MAGIC     f"https://{host}/serving-endpoints/marvel-character-endpoint/invocations",
# MAGIC     headers={"Authorization": f"Bearer {token}"},
# MAGIC     json={
# MAGIC         "dataframe_records": [{
# MAGIC             "APPEARANCES": 50,
# MAGIC             "Year": 1990,
# MAGIC             "ALIGN": "Good Characters",
# MAGIC             "EYE": "Blue Eyes",
# MAGIC             "HAIR": "Brown Hair",
# MAGIC             "SEX": "Male Characters",
# MAGIC             "GSM": "",
# MAGIC             "ALIVE": "Living Characters"
# MAGIC         }]
# MAGIC     }
# MAGIC )
# MAGIC 
# MAGIC print(response.json())
# MAGIC # Output: {"predictions": ["alive"]} or {"predictions": [1]}
# MAGIC ```
