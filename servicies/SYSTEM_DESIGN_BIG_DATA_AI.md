# System Design for Big Data & AI - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Big Data Architecture Patterns](#big-data-architecture-patterns)
3. [ML/AI System Design](#mlai-system-design)
4. [Data Processing Patterns](#data-processing-patterns)
5. [Data Pipeline Architectures](#data-pipeline-architectures)
6. [Model Serving Patterns](#model-serving-patterns)
7. [Scalability Patterns](#scalability-patterns)
8. [Database Selection Guide](#database-selection-guide)
9. [Caching Strategies](#caching-strategies)
10. [Message Queue Patterns](#message-queue-patterns)
11. [Real-World Case Studies](#real-world-case-studies)
12. [Complete Architecture Examples](#complete-architecture-examples)

---

## Introduction

### What is Big Data?

**The 5 Vs of Big Data:**
- **Volume** - Petabytes of data
- **Velocity** - High-speed data ingestion
- **Variety** - Structured, unstructured, semi-structured
- **Veracity** - Data quality and trustworthiness
- **Value** - Insights from data

### Big Data vs Traditional Systems

| Aspect | Traditional | Big Data |
|--------|------------|----------|
| **Data Size** | GB to TB | TB to PB |
| **Processing** | Vertical scaling | Horizontal scaling |
| **Storage** | RDBMS | Distributed storage (HDFS, S3) |
| **Latency** | Real-time | Batch + Real-time |
| **Schema** | Schema-first | Schema-on-read |

---

## Big Data Architecture Patterns

### 1. Lambda Architecture

**Concept:** Combine batch and real-time processing for complete view

```
                  ┌─────────────────┐
                  │   Data Source   │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        Batch Layer              Speed Layer
              │                         │
      ┌───────┴───────┐         ┌──────┴──────┐
      │  Batch Views  │         │ Real-time   │
      │  (Hadoop/     │         │ Views       │
      │   Spark)      │         │ (Storm/     │
      └───────┬───────┘         │  Flink)     │
              │                 └──────┬──────┘
              │                        │
              └────────────┬───────────┘
                           │
                    ┌──────┴──────┐
                    │ Serving     │
                    │ Layer       │
                    │ (Query)     │
                    └─────────────┘
```

**Implementation:**

```python
# Batch Layer - Process historical data
from pyspark.sql import SparkSession

class BatchProcessor:
    """
    Batch Layer: Process all historical data
    - High latency (hours)
    - High accuracy
    - Complete data view
    """

    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("BatchProcessor") \
            .getOrCreate()

    def process_daily_pageviews(self, date: str):
        """
        Process all pageviews for a given date
        Example: Count pageviews per URL
        """
        # Read from data lake (S3, HDFS)
        df = self.spark.read.parquet(f"s3://data-lake/pageviews/date={date}")

        # Aggregate
        pageview_counts = df.groupBy("url").count()

        # Write to batch view (HBase, Cassandra)
        pageview_counts.write \
            .mode("overwrite") \
            .format("org.apache.hadoop.hbase.spark") \
            .option("hbase.table", "pageview_counts_batch") \
            .save()

        return pageview_counts

# Speed Layer - Process real-time data
from kafka import KafkaConsumer
import redis

class SpeedProcessor:
    """
    Speed Layer: Process incoming data in real-time
    - Low latency (seconds)
    - Approximate accuracy
    - Only recent data
    """

    def __init__(self):
        self.consumer = KafkaConsumer(
            'pageviews',
            bootstrap_servers=['localhost:9092'],
            group_id='speed-processor'
        )
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    def process_realtime_pageviews(self):
        """
        Process pageviews in real-time
        Update counts in Redis
        """
        for message in self.consumer:
            pageview = json.loads(message.value)

            url = pageview['url']
            timestamp = pageview['timestamp']

            # Increment real-time counter
            key = f"pageview:realtime:{url}"
            self.redis.incr(key)

            # Set expiration (data will be in batch layer eventually)
            self.redis.expire(key, 86400)  # 24 hours

# Serving Layer - Merge batch + real-time views
class ServingLayer:
    """
    Serving Layer: Merge batch and speed layer results
    """

    def __init__(self):
        self.hbase = HBaseClient()
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    def get_pageview_count(self, url: str) -> int:
        """
        Get total pageview count (batch + real-time)
        """
        # Get batch count (historical)
        batch_count = self.hbase.get("pageview_counts_batch", url)

        # Get real-time count (recent)
        realtime_count = int(self.redis.get(f"pageview:realtime:{url}") or 0)

        # Merge
        total_count = batch_count + realtime_count

        return total_count
```

**Advantages:**
✅ Complete data view (batch + real-time)
✅ Fault-tolerant (batch layer can recompute)
✅ Handles high throughput

**Disadvantages:**
❌ Complex (two processing pipelines)
❌ Duplicate code (batch + speed logic)
❌ Higher maintenance

**When to Use:**
✅ Need both historical analysis and real-time updates
✅ Can tolerate eventual consistency
✅ High data volume

---

### 2. Kappa Architecture

**Concept:** Everything is a stream. No batch layer!

```
                  ┌─────────────────┐
                  │   Data Source   │
                  └────────┬────────┘
                           │
                  ┌────────┴────────┐
                  │  Stream Layer   │
                  │  (Kafka/Flink)  │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────┴──────┐           ┌─────┴──────┐
        │ Real-time  │           │ Historical │
        │ Storage    │           │ Storage    │
        │ (Redis)    │           │ (S3/HDFS)  │
        └────────────┘           └────────────┘
```

**Implementation:**

```python
from kafka import KafkaConsumer, KafkaProducer
from flink.streaming import StreamExecutionEnvironment

class KappaArchitecture:
    """
    Kappa Architecture: Stream-first approach
    - Everything is a stream
    - Replay stream for historical data
    - Simpler than Lambda
    """

    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()

    def process_pageviews(self):
        """
        Process pageviews as a continuous stream
        """
        # Read from Kafka
        stream = self.env.add_source(
            FlinkKafkaConsumer(
                topics=['pageviews'],
                deserialization_schema=JsonSchema(),
                properties={'bootstrap.servers': 'localhost:9092'}
            )
        )

        # Window aggregation (e.g., 1-hour windows)
        windowed_counts = stream \
            .key_by(lambda x: x['url']) \
            .time_window(Time.hours(1)) \
            .reduce(lambda a, b: {'url': a['url'], 'count': a['count'] + b['count']})

        # Write to multiple sinks
        windowed_counts.add_sink(RedisSink())  # Real-time queries
        windowed_counts.add_sink(S3Sink())     # Historical storage

        # Execute
        self.env.execute("PageviewProcessor")

    def reprocess_historical_data(self, start_offset: int):
        """
        Reprocess historical data by replaying Kafka stream
        """
        consumer = KafkaConsumer(
            'pageviews',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',  # Start from beginning
            enable_auto_commit=False
        )

        # Seek to specific offset
        consumer.seek(TopicPartition('pageviews', 0), start_offset)

        # Process same logic as real-time
        for message in consumer:
            # Same processing logic
            pass
```

**Advantages:**
✅ Simpler than Lambda (one codebase)
✅ Easy to reprocess data (replay stream)
✅ Natural for event-driven systems

**Disadvantages:**
❌ Requires durable message queue (Kafka)
❌ Reprocessing can be slow for large datasets
❌ More complex stream processing

**When to Use:**
✅ Event-driven systems
✅ Need to reprocess data frequently
✅ Simpler architecture preferred

---

### 3. Data Lake Architecture

**Concept:** Store all raw data, process on-demand

```
        ┌──────────────────────────────────┐
        │         Data Sources             │
        │  (APIs, DBs, Logs, IoT)          │
        └─────────────┬────────────────────┘
                      │
          ┌───────────┴───────────┐
          │   Data Ingestion      │
          │  (Kafka, Flume,       │
          │   Airflow)            │
          └───────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │      Data Lake        │
          │   (S3, HDFS, ADLS)    │
          │                       │
          │  Raw → Processed →    │
          │  Curated              │
          └───────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │   Processing Layer    │
          │  (Spark, Databricks)  │
          └───────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │    Serving Layer      │
          │  (Data Warehouse,     │
          │   Analytics Tools)    │
          └───────────────────────┘
```

**Zones in Data Lake:**

```python
# Data Lake Structure
data-lake/
├── bronze/           # Raw data (as-is)
│   ├── logs/
│   ├── api_data/
│   └── databases/
├── silver/           # Cleaned & validated
│   ├── cleaned_logs/
│   └── validated_data/
└── gold/             # Business-level aggregates
    ├── user_metrics/
    ├── product_metrics/
    └── reports/

# Implementation
import boto3
from pyspark.sql import SparkSession

class DataLakeProcessor:
    """
    Data Lake: Multi-zone processing
    Bronze (Raw) → Silver (Cleaned) → Gold (Aggregated)
    """

    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.s3 = boto3.client('s3')

    def ingest_to_bronze(self, source_data, table_name: str):
        """
        Stage 1: Ingest raw data to Bronze zone
        - No transformations
        - Keep original format
        - Partition by date
        """
        bronze_path = f"s3://data-lake/bronze/{table_name}/date={datetime.now().date()}"

        source_data.write \
            .mode("append") \
            .parquet(bronze_path)

    def bronze_to_silver(self, table_name: str):
        """
        Stage 2: Clean and validate data
        - Remove duplicates
        - Fix data types
        - Validate schemas
        - Handle nulls
        """
        # Read bronze data
        bronze_df = self.spark.read.parquet(f"s3://data-lake/bronze/{table_name}")

        # Clean data
        silver_df = bronze_df \
            .dropDuplicates() \
            .filter("user_id IS NOT NULL") \
            .withColumn("timestamp", col("timestamp").cast("timestamp")) \
            .withColumn("amount", col("amount").cast("decimal(10,2)"))

        # Write to silver
        silver_path = f"s3://data-lake/silver/{table_name}"
        silver_df.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(silver_path)

    def silver_to_gold(self, aggregation_type: str):
        """
        Stage 3: Create business-level aggregates
        - User metrics
        - Product metrics
        - Reports
        """
        silver_df = self.spark.read.parquet("s3://data-lake/silver/transactions")

        if aggregation_type == "daily_revenue":
            gold_df = silver_df \
                .groupBy("date") \
                .agg(
                    sum("amount").alias("total_revenue"),
                    count("*").alias("transaction_count"),
                    avg("amount").alias("avg_transaction")
                )

            gold_path = "s3://data-lake/gold/daily_revenue"
            gold_df.write.mode("overwrite").parquet(gold_path)
```

**Advantages:**
✅ Store all data (cheap storage)
✅ Schema-on-read flexibility
✅ Supports multiple analytics tools

**Disadvantages:**
❌ Can become "data swamp" if not managed
❌ Governance challenges
❌ Query performance can be slow

**When to Use:**
✅ Need to store diverse data types
✅ Exploratory data analysis
✅ Future use cases unknown

---

## ML/AI System Design

### ML System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     ML System Architecture                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Sources → Feature Store → Model Training              │
│                                        ↓                     │
│                                  Model Registry              │
│                                        ↓                     │
│                              Model Serving (Inference)       │
│                                        ↓                     │
│                                  Monitoring & Feedback       │
└──────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Feature Store

**Purpose:** Centralized storage for ML features

```python
# feature_store.py
import feast
from datetime import datetime, timedelta

class FeatureStore:
    """
    Feature Store: Centralized feature management
    - Consistent features for training and inference
    - Feature versioning
    - Feature discovery
    """

    def __init__(self):
        self.store = feast.FeatureStore(repo_path="feature_repo/")

    def define_user_features(self):
        """Define user features"""
        from feast import Entity, Feature, FeatureView, FileSource, ValueType

        # Define entity (user)
        user = Entity(
            name="user_id",
            value_type=ValueType.INT64,
            description="User ID"
        )

        # Define feature source
        user_source = FileSource(
            path="data/user_features.parquet",
            event_timestamp_column="event_timestamp",
        )

        # Define feature view
        user_features = FeatureView(
            name="user_features",
            entities=["user_id"],
            ttl=timedelta(days=1),
            features=[
                Feature(name="age", dtype=ValueType.INT64),
                Feature(name="total_purchases", dtype=ValueType.INT64),
                Feature(name="avg_purchase_amount", dtype=ValueType.DOUBLE),
                Feature(name="days_since_last_purchase", dtype=ValueType.INT64),
            ],
            online=True,
            batch_source=user_source,
        )

        return user_features

    def get_online_features(self, user_ids: list):
        """
        Get features for real-time inference
        """
        feature_vector = self.store.get_online_features(
            features=[
                "user_features:age",
                "user_features:total_purchases",
                "user_features:avg_purchase_amount",
                "user_features:days_since_last_purchase",
            ],
            entity_rows=[{"user_id": user_id} for user_id in user_ids]
        )

        return feature_vector.to_df()

    def get_historical_features(self, entity_df):
        """
        Get features for training (point-in-time correct)
        """
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=[
                "user_features:age",
                "user_features:total_purchases",
                "user_features:avg_purchase_amount",
                "user_features:days_since_last_purchase",
            ]
        ).to_df()

        return training_df
```

#### 2. Model Training Pipeline

```python
# training_pipeline.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class ModelTrainingPipeline:
    """
    Model Training Pipeline:
    - Load data from feature store
    - Train model
    - Log to model registry
    - Evaluate performance
    """

    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        mlflow.set_tracking_uri("http://localhost:5000")

    def train_churn_prediction_model(self):
        """
        Train a customer churn prediction model
        """
        # Start MLflow run
        with mlflow.start_run(run_name="churn_prediction_v1"):

            # 1. Load features from feature store
            entity_df = pd.DataFrame({
                "user_id": range(1, 10000),
                "event_timestamp": [datetime.now()] * 9999
            })

            features_df = self.feature_store.get_historical_features(entity_df)

            # 2. Load labels
            labels_df = pd.read_csv("data/churn_labels.csv")
            data = features_df.merge(labels_df, on="user_id")

            # 3. Split data
            X = data[['age', 'total_purchases', 'avg_purchase_amount', 'days_since_last_purchase']]
            y = data['churned']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # 4. Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)

            # 5. Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            # 6. Log to MLflow
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)

            # 7. Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="churn_prediction_model"
            )

            print(f"Model trained! Test accuracy: {test_score:.4f}")

            return model
```

#### 3. Model Registry

```python
# model_registry.py
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    """
    Model Registry: Manage model versions
    - Version control
    - Stage management (staging, production)
    - Model lineage
    """

    def __init__(self):
        self.client = MlflowClient()

    def promote_to_production(self, model_name: str, version: int):
        """
        Promote a model version to production
        """
        # Transition to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        print(f"Model {model_name} version {version} promoted to Production")

    def get_production_model(self, model_name: str):
        """
        Get current production model
        """
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    def compare_models(self, model_name: str, version1: int, version2: int):
        """
        Compare two model versions
        """
        run1 = self.client.get_model_version(model_name, version1).run_id
        run2 = self.client.get_model_version(model_name, version2).run_id

        metrics1 = self.client.get_run(run1).data.metrics
        metrics2 = self.client.get_run(run2).data.metrics

        comparison = {
            f"v{version1}": metrics1,
            f"v{version2}": metrics2
        }

        return comparison
```

#### 4. Model Serving

**Option A: Batch Inference**

```python
# batch_inference.py
from pyspark.sql import SparkSession
import mlflow

class BatchInferenceService:
    """
    Batch Inference: Process large datasets offline
    - Use case: Daily predictions for all users
    - High throughput
    - Higher latency (minutes to hours)
    """

    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.model = mlflow.sklearn.load_model("models:/churn_prediction_model/Production")

    def predict_daily_churn(self, date: str):
        """
        Generate churn predictions for all users
        """
        # Load features for all users
        features_df = self.spark.read.parquet(f"s3://features/user_features/date={date}")

        # Convert to pandas for sklearn
        features_pdf = features_df.toPandas()

        # Predict
        predictions = self.model.predict_proba(
            features_pdf[['age', 'total_purchases', 'avg_purchase_amount', 'days_since_last_purchase']]
        )

        # Add predictions to dataframe
        features_pdf['churn_probability'] = predictions[:, 1]

        # Save results
        results_df = self.spark.createDataFrame(features_pdf)
        results_df.write.mode("overwrite").parquet(f"s3://predictions/churn/date={date}")

        return results_df
```

**Option B: Real-time Inference**

```python
# realtime_inference.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI()

# Load model at startup
model = mlflow.sklearn.load_model("models:/churn_prediction_model/Production")
feature_store = FeatureStore()

class PredictionRequest(BaseModel):
    user_id: int

class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    high_risk: bool

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(request: PredictionRequest):
    """
    Real-time churn prediction
    - Low latency (<100ms)
    - Single prediction
    """
    # Get features from feature store
    features = feature_store.get_online_features([request.user_id])

    # Predict
    X = features[['age', 'total_purchases', 'avg_purchase_amount', 'days_since_last_purchase']].values
    churn_prob = model.predict_proba(X)[0, 1]

    return PredictionResponse(
        user_id=request.user_id,
        churn_probability=float(churn_prob),
        high_risk=churn_prob > 0.7
    )

# Run with: uvicorn realtime_inference:app --host 0.0.0.0 --port 8000
```

**Option C: Streaming Inference**

```python
# streaming_inference.py
from kafka import KafkaConsumer, KafkaProducer
import json
import mlflow

class StreamingInferenceService:
    """
    Streaming Inference: Process events in real-time
    - Use case: Fraud detection, real-time recommendations
    - Medium latency (milliseconds to seconds)
    - Continuous processing
    """

    def __init__(self):
        self.consumer = KafkaConsumer(
            'user_events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.model = mlflow.sklearn.load_model("models:/churn_prediction_model/Production")
        self.feature_store = FeatureStore()

    def process_events(self):
        """
        Process user events and make predictions
        """
        for message in self.consumer:
            event = message.value

            user_id = event['user_id']
            event_type = event['event_type']

            # Get latest features
            features = self.feature_store.get_online_features([user_id])

            # Predict churn
            X = features[['age', 'total_purchases', 'avg_purchase_amount', 'days_since_last_purchase']].values
            churn_prob = self.model.predict_proba(X)[0, 1]

            # If high risk, send alert
            if churn_prob > 0.7:
                alert = {
                    'user_id': user_id,
                    'churn_probability': float(churn_prob),
                    'timestamp': datetime.now().isoformat(),
                    'action': 'send_retention_offer'
                }

                self.producer.send('churn_alerts', value=alert)

            print(f"Processed user {user_id}: churn_prob={churn_prob:.4f}")
```

#### 5. Model Monitoring

```python
# model_monitoring.py
import pandas as pd
from prometheus_client import Gauge, Counter, Histogram
from scipy import stats

class ModelMonitor:
    """
    Model Monitoring: Track model performance in production
    - Data drift detection
    - Model performance metrics
    - Alerting
    """

    def __init__(self):
        # Prometheus metrics
        self.prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')
        self.prediction_count = Counter('model_prediction_total', 'Total predictions')
        self.data_drift_score = Gauge('model_data_drift_score', 'Data drift score')

    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Detect data drift using KS test
        """
        drift_scores = {}

        for column in reference_data.columns:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                reference_data[column],
                current_data[column]
            )

            drift_scores[column] = {
                'statistic': statistic,
                'p_value': p_value,
                'drifted': p_value < 0.05  # Significant drift if p < 0.05
            }

            # Update Prometheus metric
            self.data_drift_score.set(statistic)

            if drift_scores[column]['drifted']:
                print(f"⚠️ Data drift detected in {column}! p-value: {p_value:.4f}")

        return drift_scores

    def track_prediction_performance(self, predictions: list, actuals: list):
        """
        Track prediction accuracy over time
        """
        accuracy = sum([p == a for p, a in zip(predictions, actuals)]) / len(predictions)

        # Log to monitoring system
        print(f"Current accuracy: {accuracy:.4f}")

        # Alert if accuracy drops below threshold
        if accuracy < 0.7:
            self.send_alert(f"Model accuracy dropped to {accuracy:.4f}")

    def send_alert(self, message: str):
        """Send alert to Slack/PagerDuty/etc."""
        # Integration with alerting system
        print(f"🚨 ALERT: {message}")
```

---

## Data Processing Patterns

### Batch vs Stream vs Micro-Batch

| Pattern | Latency | Throughput | Use Case |
|---------|---------|------------|----------|
| **Batch** | Hours | Very High | Daily reports, ETL |
| **Micro-Batch** | Seconds | High | Near real-time analytics |
| **Stream** | Milliseconds | Medium | Fraud detection, real-time alerts |

### Batch Processing

```python
# batch_processing.py
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

class BatchProcessor:
    """
    Batch Processing: Process large datasets in scheduled jobs
    - High throughput
    - High latency
    - Cost-effective
    """

    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("BatchProcessor") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "4") \
            .getOrCreate()

    def process_daily_logs(self, date: str):
        """
        Process a full day of logs
        Example: Calculate user engagement metrics
        """
        # Read logs for specific date
        logs_df = self.spark.read.json(f"s3://logs/app_logs/date={date}")

        # Aggregate user engagement
        user_metrics = logs_df.groupBy("user_id").agg(
            count("*").alias("total_events"),
            countDistinct("session_id").alias("sessions"),
            sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
            sum("revenue").alias("total_revenue")
        )

        # Calculate derived metrics
        user_metrics = user_metrics.withColumn(
            "events_per_session",
            col("total_events") / col("sessions")
        )

        # Write results
        user_metrics.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(f"s3://metrics/user_engagement/date={date}")

        return user_metrics

    def backfill_data(self, start_date: str, end_date: str):
        """
        Backfill historical data
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing {date_str}...")

            self.process_daily_logs(date_str)

            current_date += timedelta(days=1)
```

### Stream Processing

```python
# stream_processing.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

class StreamProcessor:
    """
    Stream Processing: Process data in real-time
    - Low latency
    - Continuous processing
    - Stateful operations
    """

    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("StreamProcessor") \
            .getOrCreate()

    def process_clickstream(self):
        """
        Process clickstream in real-time
        Calculate real-time metrics with windowing
        """
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "clickstream") \
            .load()

        # Parse JSON
        clicks_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")

        # Windowed aggregation (5-minute tumbling windows)
        windowed_counts = clicks_df \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy(
                window("timestamp", "5 minutes"),
                "page_url"
            ) \
            .agg(
                count("*").alias("pageviews"),
                countDistinct("user_id").alias("unique_users")
            )

        # Write to multiple sinks
        query = windowed_counts \
            .writeStream \
            .outputMode("update") \
            .format("console") \
            .start()

        query.awaitTermination()

    def detect_fraud_realtime(self):
        """
        Real-time fraud detection using stateful stream processing
        """
        transactions_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "transactions") \
            .load()

        # Parse transactions
        parsed_df = transactions_df.select(
            from_json(col("value").cast("string"), transaction_schema).alias("txn")
        ).select("txn.*")

        # Stateful processing: Track transactions per user in 1-hour window
        fraud_detection = parsed_df \
            .withWatermark("timestamp", "2 hours") \
            .groupBy(
                window("timestamp", "1 hour"),
                "user_id"
            ) \
            .agg(
                count("*").alias("txn_count"),
                sum("amount").alias("total_amount"),
                countDistinct("merchant_id").alias("unique_merchants")
            ) \
            .filter(
                (col("txn_count") > 10) |  # More than 10 transactions
                (col("total_amount") > 10000) |  # Total > $10k
                (col("unique_merchants") > 5)  # Too many merchants
            )

        # Write alerts to Kafka
        query = fraud_detection \
            .selectExpr("to_json(struct(*)) AS value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "fraud_alerts") \
            .start()

        query.awaitTermination()
```

---

## Data Pipeline Architectures

### ETL vs ELT

**ETL (Extract, Transform, Load):**
```python
# etl_pipeline.py
class ETLPipeline:
    """
    ETL: Transform data before loading to warehouse
    - Good for: Small to medium data
    - Transform in processing layer
    """

    def extract(self, source_db: str):
        """Extract data from source"""
        conn = psycopg2.connect(source_db)
        df = pd.read_sql("SELECT * FROM users", conn)
        return df

    def transform(self, df: pd.DataFrame):
        """Transform data"""
        # Clean
        df = df.dropna()

        # Enrich
        df['full_name'] = df['first_name'] + ' ' + df['last_name']

        # Aggregate
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100])

        return df

    def load(self, df: pd.DataFrame, target_db: str):
        """Load to data warehouse"""
        engine = create_engine(target_db)
        df.to_sql('users_transformed', engine, if_exists='replace')
```

**ELT (Extract, Load, Transform):**
```python
# elt_pipeline.py
class ELTPipeline:
    """
    ELT: Load raw data first, transform in warehouse
    - Good for: Big data
    - Use warehouse's compute power
    """

    def extract_and_load(self, source_db: str, data_lake: str):
        """Extract and load raw data directly"""
        conn = psycopg2.connect(source_db)
        df = pd.read_sql("SELECT * FROM users", conn)

        # Load raw to S3/Data Lake
        df.to_parquet(f"{data_lake}/raw/users.parquet")

    def transform_in_warehouse(self):
        """Transform using Spark/warehouse SQL"""
        spark = SparkSession.builder.getOrCreate()

        # Read raw data
        df = spark.read.parquet("s3://data-lake/raw/users.parquet")

        # Transform using Spark (leveraging cluster)
        transformed_df = df \
            .dropna() \
            .withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name"))) \
            .withColumn("age_group",
                when(col("age") < 18, "0-18")
                .when(col("age") < 35, "18-35")
                .when(col("age") < 50, "35-50")
                .otherwise("50+")
            )

        # Write to warehouse
        transformed_df.write.mode("overwrite").saveAsTable("users_transformed")
```

### Orchestration with Apache Airflow

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_user_metrics',
    default_args=default_args,
    description='Calculate daily user engagement metrics',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False
)

# Task 1: Extract data from production DB
def extract_data(**context):
    execution_date = context['execution_date']
    # Extract logic
    print(f"Extracting data for {execution_date}")

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

# Task 2: Process with Spark
spark_task = SparkSubmitOperator(
    task_id='process_with_spark',
    application='/path/to/batch_processing.py',
    conn_id='spark_default',
    dag=dag
)

# Task 3: Load to data warehouse
def load_to_warehouse(**context):
    # Load logic
    print("Loading to warehouse")

load_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag
)

# Task 4: Run data quality checks
def data_quality_checks(**context):
    # Quality checks
    print("Running quality checks")

quality_task = PythonOperator(
    task_id='data_quality_checks',
    python_callable=data_quality_checks,
    dag=dag
)

# Define dependencies
extract_task >> spark_task >> load_task >> quality_task
```

---

## Model Serving Patterns

### Comparison Table

| Pattern | Latency | Throughput | Cost | Use Case |
|---------|---------|------------|------|----------|
| **Batch** | Hours | Very High | Low | Daily predictions |
| **Real-time API** | <100ms | Low-Medium | High | User-facing predictions |
| **Streaming** | Seconds | High | Medium | Event-driven predictions |
| **Edge** | <10ms | N/A | Low | On-device inference |

### Pattern Selection

```python
def choose_serving_pattern(requirements: dict) -> str:
    """
    Decision tree for choosing model serving pattern
    """
    latency_required = requirements.get('latency_ms')
    throughput = requirements.get('requests_per_second')
    budget = requirements.get('budget')

    if latency_required > 3600000:  # > 1 hour
        return "Batch Processing"

    elif latency_required < 100:  # < 100ms
        if budget == 'high':
            return "Real-time API (Kubernetes + Autoscaling)"
        else:
            return "Edge Inference"

    elif 100 <= latency_required <= 1000:  # 100ms - 1s
        return "Streaming Inference (Kafka + Flink)"

    else:
        return "Micro-batch (Spark Streaming)"
```

---

## Scalability Patterns

### Horizontal Scaling

```python
# kubernetes_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 3  # Start with 3 replicas
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-api
        image: myregistry/model-serving:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```python
# nginx.conf
upstream model_servers {
    least_conn;  # Use least connections algorithm

    server model-server-1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server model-server-2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server model-server-3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;

    location /predict {
        proxy_pass http://model_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

### Caching for ML

```python
# model_cache.py
import redis
import hashlib
import json

class ModelPredictionCache:
    """
    Cache predictions to reduce model inference calls
    """

    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = 3600  # 1 hour

    def get_prediction(self, features: dict):
        """
        Get cached prediction or compute new one
        """
        # Create cache key from features
        cache_key = self._create_key(features)

        # Check cache
        cached_result = self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        # Cache miss - compute prediction
        prediction = self.model.predict([list(features.values())])[0]

        # Cache result
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps({"prediction": float(prediction)})
        )

        return {"prediction": float(prediction)}

    def _create_key(self, features: dict) -> str:
        """Create deterministic cache key from features"""
        feature_str = json.dumps(features, sort_keys=True)
        return f"prediction:{hashlib.md5(feature_str.encode()).hexdigest()}"
```

---

## Database Selection Guide

### Decision Matrix

```python
class DatabaseSelector:
    """
    Decision tree for choosing the right database
    """

    @staticmethod
    def select_database(requirements: dict) -> str:
        data_structure = requirements.get('data_structure')
        read_write_ratio = requirements.get('read_write_ratio')
        consistency = requirements.get('consistency')
        scale = requirements.get('scale')
        query_pattern = requirements.get('query_pattern')

        # Relational data
        if data_structure == 'relational':
            if scale == 'small':
                return "PostgreSQL"
            elif consistency == 'strong':
                return "PostgreSQL with read replicas"
            else:
                return "CockroachDB (distributed SQL)"

        # Document data
        elif data_structure == 'document':
            if scale == 'large':
                return "MongoDB (sharded)"
            else:
                return "MongoDB"

        # Time-series data
        elif data_structure == 'time_series':
            if scale == 'large':
                return "InfluxDB or TimescaleDB"
            else:
                return "TimescaleDB"

        # Key-value (caching)
        elif data_structure == 'key_value':
            if requirements.get('persistence') == True:
                return "Redis with AOF"
            else:
                return "Redis or Memcached"

        # Graph data
        elif data_structure == 'graph':
            return "Neo4j"

        # Wide-column (big data)
        elif data_structure == 'wide_column':
            return "Cassandra or HBase"

        # Search
        elif query_pattern == 'full_text_search':
            return "Elasticsearch"

        # Analytics/OLAP
        elif query_pattern == 'analytical':
            return "ClickHouse or Apache Druid"

        return "PostgreSQL (default)"
```

### Database Comparison for Big Data

| Database | Best For | Max Scale | Query Speed | Consistency |
|----------|----------|-----------|-------------|-------------|
| **PostgreSQL** | Relational, ACID | TB | Fast | Strong |
| **Cassandra** | Time-series, IoT | PB | Very Fast (writes) | Eventual |
| **MongoDB** | Documents, JSON | TB | Fast | Tunable |
| **ClickHouse** | Analytics, OLAP | PB | Very Fast (reads) | Eventual |
| **Elasticsearch** | Search, logs | TB | Very Fast (search) | Eventual |
| **Redis** | Caching, sessions | GB | Extremely Fast | Eventual |

---

## Caching Strategies

### Multi-Level Caching

```python
# multi_level_cache.py
import redis
from functools import lru_cache

class MultiLevelCache:
    """
    Multi-level caching strategy:
    L1: In-memory (LRU)
    L2: Redis
    L3: Database
    """

    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.db = get_database_connection()

    @lru_cache(maxsize=1000)  # L1: In-memory cache
    def get_user_data(self, user_id: int):
        """
        Get user data with multi-level caching
        """
        # L1: Check in-memory cache (handled by @lru_cache decorator)

        # L2: Check Redis
        cache_key = f"user:{user_id}"
        cached_data = self.redis.get(cache_key)

        if cached_data:
            return json.loads(cached_data)

        # L3: Query database
        user_data = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

        if user_data:
            # Populate caches
            user_json = json.dumps(user_data)

            # L2: Store in Redis (TTL: 1 hour)
            self.redis.setex(cache_key, 3600, user_json)

            # L1: Already cached by @lru_cache

        return user_data
```

### Cache Invalidation Patterns

```python
# cache_invalidation.py
class CacheInvalidationPatterns:
    """
    Different cache invalidation strategies
    """

    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)

    # Pattern 1: TTL (Time-To-Live)
    def set_with_ttl(self, key: str, value: str, ttl_seconds: int = 3600):
        """Cache expires automatically after TTL"""
        self.redis.setex(key, ttl_seconds, value)

    # Pattern 2: Write-Through
    def write_through_update(self, user_id: int, data: dict):
        """Update DB and cache simultaneously"""
        # Update database
        self.db.update(user_id, data)

        # Update cache
        cache_key = f"user:{user_id}"
        self.redis.set(cache_key, json.dumps(data))

    # Pattern 3: Cache Aside (Lazy Loading)
    def cache_aside_get(self, user_id: int):
        """Load to cache only when requested"""
        cache_key = f"user:{user_id}"

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Cache miss - load from DB
        data = self.db.get(user_id)

        # Populate cache
        self.redis.setex(cache_key, 3600, json.dumps(data))

        return data

    # Pattern 4: Event-Driven Invalidation
    def invalidate_on_event(self, event_type: str, user_id: int):
        """Invalidate cache when specific event occurs"""
        if event_type == "user_updated":
            cache_key = f"user:{user_id}"
            self.redis.delete(cache_key)

            # Also invalidate related caches
            self.redis.delete(f"user_profile:{user_id}")
            self.redis.delete(f"user_preferences:{user_id}")
```

---

## Message Queue Patterns

### Pub/Sub Pattern

```python
# pubsub_pattern.py
import pika
import json

class EventBus:
    """
    Pub/Sub Pattern using RabbitMQ
    - Publishers don't know about subscribers
    - Multiple subscribers can receive same message
    """

    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        # Declare topic exchange
        self.channel.exchange_declare(exchange='events', exchange_type='topic')

    def publish(self, event_type: str, event_data: dict):
        """
        Publish event to all subscribers
        """
        self.channel.basic_publish(
            exchange='events',
            routing_key=event_type,  # e.g., 'user.created', 'order.shipped'
            body=json.dumps(event_data)
        )

    def subscribe(self, event_pattern: str, callback):
        """
        Subscribe to events matching pattern
        Pattern examples:
        - 'user.*' - All user events
        - '*.created' - All created events
        - 'order.#' - All order events
        """
        result = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        self.channel.queue_bind(
            exchange='events',
            queue=queue_name,
            routing_key=event_pattern
        )

        def wrapper(ch, method, properties, body):
            event_data = json.loads(body)
            callback(event_data)

        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=wrapper,
            auto_ack=True
        )

        self.channel.start_consuming()

# Usage
event_bus = EventBus()

# Publisher
event_bus.publish('user.created', {'user_id': 123, 'email': 'test@example.com'})

# Subscriber 1: Email service
def send_welcome_email(event):
    print(f"Sending welcome email to {event['email']}")

# Subscriber 2: Analytics service
def track_user_signup(event):
    print(f"Tracking signup for user {event['user_id']}")

# Subscribe
event_bus.subscribe('user.created', send_welcome_email)
event_bus.subscribe('user.*', track_user_signup)
```

### Work Queue Pattern

```python
# work_queue_pattern.py
from kafka import KafkaConsumer, KafkaProducer
import json

class TaskQueue:
    """
    Work Queue Pattern using Kafka
    - Distribute tasks among workers
    - Load balancing
    - Competing consumers
    """

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def submit_task(self, task_type: str, task_data: dict):
        """
        Submit task to queue
        """
        task = {
            'task_type': task_type,
            'data': task_data,
            'timestamp': datetime.now().isoformat()
        }

        self.producer.send('tasks', value=task)

class Worker:
    """
    Worker that processes tasks from queue
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.consumer = KafkaConsumer(
            'tasks',
            bootstrap_servers=['localhost:9092'],
            group_id='task-workers',  # Same group = competing consumers
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def start(self):
        """
        Start processing tasks
        """
        print(f"Worker {self.worker_id} started")

        for message in self.consumer:
            task = message.value
            self.process_task(task)

    def process_task(self, task: dict):
        """
        Process individual task
        """
        print(f"Worker {self.worker_id} processing {task['task_type']}")

        if task['task_type'] == 'send_email':
            self.send_email(task['data'])
        elif task['task_type'] == 'generate_report':
            self.generate_report(task['data'])

        print(f"Worker {self.worker_id} completed {task['task_type']}")

# Usage: Run multiple workers for load balancing
worker1 = Worker(worker_id=1)
worker2 = Worker(worker_id=2)
worker3 = Worker(worker_id=3)

# Each worker processes tasks from the same queue
# Kafka automatically distributes tasks among workers
```

---

## Real-World Case Studies

### Case Study 1: Netflix Recommendation System

**Architecture:**

```
┌────────────────────────────────────────────────────────┐
│              Netflix Recommendation System             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  User Events → Kafka → Flink (Stream Processing)      │
│                          ↓                             │
│                    Feature Store                       │
│                          ↓                             │
│  Batch Training (Spark) ←→ Model Registry (MLflow)    │
│                          ↓                             │
│                   Model Serving                        │
│                  (Real-time + Batch)                   │
│                          ↓                             │
│                  Cassandra (Storage)                   │
│                          ↓                             │
│                     CDN (Cache)                        │
└────────────────────────────────────────────────────────┘
```

**Key Technologies:**
- **Kafka** - Event streaming (view events, ratings)
- **Flink** - Real-time feature computation
- **Spark** - Batch model training
- **Cassandra** - Recommendations storage (billions of rows)
- **EVCache** (Redis-based) - Low-latency cache

**Architecture Decisions:**
1. **Hybrid approach** - Real-time + batch predictions
2. **Pre-compute** - Generate top-N recommendations offline
3. **Personalize** - Real-time adjustment based on current session
4. **A/B testing** - Multiple models served simultaneously

---

### Case Study 2: Uber Real-Time Pricing

**Architecture:**

```
┌────────────────────────────────────────────────────────┐
│               Uber Surge Pricing System                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Ride Requests → Kafka → Stream Processing            │
│                            ↓                           │
│                   Demand Calculation                   │
│                            ↓                           │
│  Supply Data (Drivers) → Geospatial Processing        │
│                            ↓                           │
│                    Pricing ML Model                    │
│                            ↓                           │
│                     Redis (Cache)                      │
│                            ↓                           │
│                    Mobile Apps (API)                   │
└────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# uber_surge_pricing.py
from kafka import KafkaConsumer
import redis
from geopy.distance import geodesic

class SurgePricingEngine:
    """
    Real-time surge pricing calculation
    """

    def __init__(self):
        self.consumer = KafkaConsumer('ride_requests', ...)
        self.redis = redis.Redis(...)
        self.pricing_model = load_model('surge_pricing_v2')

    def calculate_surge(self, location: tuple, timestamp: datetime):
        """
        Calculate surge multiplier for a location
        """
        lat, lon = location

        # Get demand in area (from stream processing)
        demand = self.get_demand_in_area(lat, lon, radius_km=2)

        # Get supply (available drivers)
        supply = self.get_available_drivers(lat, lon, radius_km=2)

        # Calculate demand/supply ratio
        ratio = demand / max(supply, 1)

        # Features for ML model
        features = {
            'demand_supply_ratio': ratio,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'weather_condition': self.get_weather(lat, lon),
            'is_event_nearby': self.check_events(lat, lon)
        }

        # Predict surge multiplier
        surge_multiplier = self.pricing_model.predict([list(features.values())])[0]

        # Cache result (TTL: 2 minutes)
        cache_key = f"surge:{lat:.4f}:{lon:.4f}"
        self.redis.setex(cache_key, 120, str(surge_multiplier))

        return surge_multiplier
```

**Key Patterns:**
- **Stream processing** - Real-time demand calculation
- **Geospatial indexing** - Fast location queries
- **Caching** - Redis for low-latency lookups
- **ML model** - Predict surge based on features

---

### Case Study 3: Twitter Timeline Generation

**Architecture:**

```
┌────────────────────────────────────────────────────────┐
│              Twitter Timeline Generation               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  New Tweet → Fanout Service                            │
│                ↓                                       │
│      ┌─────────┴─────────┐                            │
│      │                   │                            │
│  Write to    Push to Redis (Timeline Cache)           │
│  Cassandra   (for active users)                       │
│      │                   │                            │
│      └─────────┬─────────┘                            │
│                ↓                                       │
│        Timeline Request                                │
│                ↓                                       │
│  Cache Hit? → Return from Redis                       │
│      │                                                 │
│  Cache Miss → Build from Cassandra + Ranking          │
└────────────────────────────────────────────────────────┘
```

**Fanout Strategies:**

```python
# twitter_fanout.py
class TimelineFanoutService:
    """
    Two fanout strategies based on user type
    """

    def __init__(self):
        self.redis = redis.Redis(...)
        self.cassandra = CassandraClient(...)

    def fanout_tweet(self, tweet: dict, author_id: int):
        """
        Distribute tweet to followers' timelines
        """
        followers = self.get_followers(author_id)
        follower_count = len(followers)

        if follower_count < 10000:
            # Push model: Pre-compute timelines (for regular users)
            self.push_fanout(tweet, followers)
        else:
            # Pull model: Compute on-demand (for celebrities)
            self.pull_fanout(tweet, author_id)

    def push_fanout(self, tweet: dict, followers: list):
        """
        Push tweet to all followers' Redis caches
        """
        for follower_id in followers:
            cache_key = f"timeline:{follower_id}"

            # Add tweet to timeline (sorted set by timestamp)
            self.redis.zadd(
                cache_key,
                {json.dumps(tweet): tweet['timestamp']}
            )

            # Keep only recent 800 tweets
            self.redis.zremrangebyrank(cache_key, 0, -801)

    def pull_fanout(self, tweet: dict, author_id: int):
        """
        Store tweet, compute timeline on request
        """
        # Just store tweet
        self.cassandra.insert_tweet(tweet)

        # Don't pre-compute (too many followers)

    def get_timeline(self, user_id: int, limit: int = 50):
        """
        Get timeline for user
        """
        cache_key = f"timeline:{user_id}"

        # Try cache first
        cached_timeline = self.redis.zrevrange(cache_key, 0, limit - 1)

        if cached_timeline:
            return [json.loads(t) for t in cached_timeline]

        # Cache miss - compute from Cassandra
        following = self.get_following(user_id)
        tweets = self.cassandra.get_recent_tweets(following, limit=1000)

        # Rank tweets
        ranked_tweets = self.rank_tweets(tweets, user_id)

        # Cache for next time
        for tweet in ranked_tweets[:800]:
            self.redis.zadd(
                cache_key,
                {json.dumps(tweet): tweet['timestamp']}
            )

        return ranked_tweets[:limit]
```

---

## Complete Architecture Examples

### Example: Real-time Fraud Detection System

```python
"""
Architecture: Real-time Fraud Detection

Components:
1. Transaction Ingestion (Kafka)
2. Stream Processing (Flink)
3. Feature Store (Redis + S3)
4. ML Model Serving (FastAPI)
5. Alerting (Slack/PagerDuty)
6. Monitoring (Prometheus/Grafana)
"""

# 1. Transaction Ingestion
from kafka import KafkaProducer

class TransactionIngestion:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def ingest_transaction(self, transaction: dict):
        self.producer.send('transactions', value=transaction)

# 2. Stream Processing
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction

class FraudDetectionProcessor:
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()

    def process(self):
        # Read from Kafka
        transactions = self.env.add_source(...)

        # Calculate features
        features = transactions.map(FeatureExtractor())

        # Score with ML model
        scored = features.map(ModelScorer())

        # Filter suspicious transactions
        alerts = scored.filter(lambda x: x['fraud_score'] > 0.8)

        # Send alerts
        alerts.add_sink(AlertSink())

        self.env.execute("FraudDetection")

class FeatureExtractor(MapFunction):
    def map(self, transaction):
        # Calculate features
        features = {
            'transaction_id': transaction['id'],
            'amount': transaction['amount'],
            'is_foreign': transaction['country'] != 'US',
            'hour_of_day': datetime.now().hour,
            # ... more features
        }
        return features

# 3. Model Serving
from fastapi import FastAPI

app = FastAPI()

@app.post("/score")
async def score_transaction(transaction: dict):
    # Get features from feature store
    features = feature_store.get_features(transaction['user_id'])

    # Score
    fraud_score = fraud_model.predict_proba([features])[0][1]

    # Alert if high risk
    if fraud_score > 0.8:
        send_alert(transaction, fraud_score)

    return {"fraud_score": float(fraud_score)}

# 4. Monitoring
from prometheus_client import Counter, Histogram

fraud_alerts = Counter('fraud_alerts_total', 'Total fraud alerts')
scoring_latency = Histogram('fraud_scoring_latency_seconds', 'Scoring latency')

@scoring_latency.time()
def score_with_monitoring(transaction):
    score = fraud_model.predict(...)
    if score > 0.8:
        fraud_alerts.inc()
    return score
```

---

## Key Takeaways

### Architecture Selection Guide

**For Big Data Processing:**
1. **Batch-heavy** → Lambda Architecture + Spark
2. **Stream-heavy** → Kappa Architecture + Flink/Kafka
3. **Flexibility** → Data Lake (Bronze-Silver-Gold)

**For ML Systems:**
1. **Training** → Feature Store + MLflow + Spark
2. **Serving** → Choose based on latency requirements
3. **Monitoring** → Always include drift detection

### Common Patterns

✅ **Use Message Queues** - Decouple services
✅ **Cache Aggressively** - Multi-level caching
✅ **Scale Horizontally** - Add more machines
✅ **Monitor Everything** - Metrics, logs, traces
✅ **Test at Scale** - Load testing is critical

### Anti-Patterns to Avoid

❌ **Premature optimization** - Start simple
❌ **No monitoring** - You can't fix what you can't see
❌ **Single point of failure** - Always have redundancy
❌ **Ignoring data quality** - Garbage in, garbage out
❌ **No versioning** - Track model/data versions

---

**Remember:** The best architecture is one that solves YOUR problem at YOUR scale! 🚀
