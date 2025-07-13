# Scalable Flight Price Prediction Using Big Data and Machine Learning

This project focuses on building a **scalable, production-grade data pipeline** to predict flight ticket prices using over **82 million rows** of historical flight booking data from Expedia.

The pipeline was developed using **Apache Spark (PySpark)** on **Google Cloud Platform (GCP)** and demonstrates the full life cycle of **Big Data Engineering and Machine Learning**, including:

- Ingesting and cleaning large-scale flight data.
- Engineering meaningful features (e.g., days until flight, seasonality, route popularity).
- Optimizing and storing the processed data in **Parquet format** on **Google Cloud Storage**.
- Training and evaluating regression models (Linear Regression, Gradient Boosted Trees) using **Spark MLlib**.
- Scaling the workflow across multiple dataset sizes using **Dataproc Spark clusters**.

## Key goals of the Project:

- **Travelers** make cost-effective booking decisions.
- **Airlines** optimize pricing strategies based on seasonality and demand trends.
- **Engineers** understand how to process and model large volumes of travel data using real-world Big Data tools.

The end result is a robust, cloud-deployed pipeline capable of handling and analyzing **massive datasets** with efficiency — demonstrating expertise in **data engineering, distributed processing, and applied machine learning**.

## Objectives

- Develop a **scalable data pipeline** to process and analyze 82 million flight records using Apache Spark on Google Cloud.
- Engineer meaningful features (e.g., days until flight, peak season, route popularity) to support flight fare prediction.
- Apply **supervised machine learning models** (Linear Regression, Gradient Boosted Trees) to predict flight prices.
- Optimize the pipeline for performance and **scalability** across varying data volumes using Dataproc clusters.
- Evaluate model accuracy using industry-standard metrics (RMSE, MAE, R²) and measure inference throughput.
- Deliver insights to help **consumers make better booking decisions** and **airlines analyze pricing trends**.

## Dataset

- **Source:** Expedia flight booking data from Kaggle  
- **Period:** April 2022 – October 2022  
- **Size:** Over **82 million rows** of detailed flight and fare records  
- **Storage:** Stored as **Parquet files** on **Google Cloud Storage (GCS)** for efficient access and processing  
- **Key Features Include:**  
  - Departure and destination airport codes  
  - Travel distance and duration  
  - Number of seats remaining  
  - Booking date vs. flight date (to calculate days until flight)  
  - Fare details: base fare, total fare, refundability, and fare basis code  
  - Flight type and class details  

The dataset provides a rich historical record of airline bookings, enabling predictive modeling that accounts for seasonal trends, route popularity, and customer behavior.

## Technologies Used

| Technology               | Role & Purpose                                         |
|--------------------------|-------------------------------------------------------|
| **Apache Spark (PySpark)**   | Distributed data processing and scalable ML model training |
| **Google Cloud Platform (GCP)** | Cloud infrastructure for storage (GCS) and compute (Dataproc)    |
| **Dataproc**                 | Managed Spark cluster for scalable big data workflows      |
| **Parquet**                  | Columnar storage format optimized for Spark processing       |
| **Jupyter Notebooks**        | Interactive development and exploratory data analysis       |
| **Spark MLlib**              | Machine learning library for regression model implementation |
| **Python**                   | Scripting, data pipeline orchestration, and model evaluation |

## Data Processing & Feature Engineering

- Removed irrelevant and duplicate columns such as `fareBasisCode` and `legId` to streamline data  
- Filtered routes to keep the top 50% by frequency, improving model focus and efficiency  
- Engineered key features including:  
  - `days_until_flight`: calculated from booking date and flight date to capture booking lead time  
  - `is_peak_season`: binary indicator for summer and Labor Day week to model seasonal demand  
  - `travelDurationMinutes`: total flight duration in minutes for fare prediction  
- Handled missing values by imputing group-based averages for continuous features  
- Encoded categorical variables (e.g., airport codes, flight type) using Spark’s `StringIndexer` and `OneHotEncoder` to prepare data for ML models  
- Stored processed data efficiently in **Parquet format** for fast access and downstream modeling

##  Exploratory Data Analysis (EDA)

- Explored route distribution to understand travel frequency across origin-destination pairs  
- Identified popular routes and high-traffic airports to inform feature engineering  
- Analyzed fare distributions to detect pricing trends, seasonality, and booking behavior  
- Performed outlier detection using **Interquartile Range (IQR)** method on fare-related columns  
- Calculated correlation between total fare, travel distance, and time until flight  
- Identified patterns between refundable and non-refundable ticket types  
- Discovered ~125 distinct routes and over **3.9 million** unique non-refundable flights

##  Machine Learning Models

This project implements two supervised regression models using **PySpark MLlib**:

### 1. Linear Regression (Baseline)
- Interpretable model for establishing baseline performance
- Fast training time on large datasets
- Helps validate feature relevance and data quality

**Performance on Full Dataset:**
- RMSE: ~127.4  
- MAE: ~81.6  
- R²: 0.553

---

### 2. Gradient Boosted Trees (Optimized)
- Captures complex non-linear relationships between features and fare
- Tuned for deeper trees and better generalization
- Scaled efficiently across dataset sizes from 10% to 100%

**Why GBT?**
- Robust to multicollinearity and missing values
- Performs significantly better on structured/tabular data
- Ideal for high-cardinality categorical features like airport codes

**MLlib Configs Used:**
- `GBTRegressor(labelCol="totalFare", featuresCol="features", maxIter=50, maxDepth=5)`
- Pipeline stages: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `GBTRegressor`

---

All models were trained using **Spark ML pipelines**, ensuring modular, scalable, and reproducible workflows.

## Model Evaluation

Both models were evaluated on prediction accuracy and scalability using standard regression metrics:

| Dataset Subset | RMSE     | MAE     | R²     | Inference Speed (preds/sec) |
|----------------|----------|---------|--------|------------------------------|
| 10%            | ~28.06   | ~81.8   | 0.549  | 3.1M+                        |
| 25%            | ~127.5   | ~81.9   | 0.550  | 17M+                         |
| 50%            | ~127.6   | ~81.8   | 0.552  | 33.9M+                       |
| 75%            | ~127.4   | ~81.7   | 0.553  | 29.8M+                       |
| 100%           | ~127.4   | ~81.6   | 0.553  | 31.8M+                       |

---

## Metrics Used:
- **Root Mean Squared Error (RMSE):** Measures prediction error magnitude  
- **Mean Absolute Error (MAE):** Captures average error per prediction  
- **R² Score:** Indicates how well the model explains variance in ticket prices  
- **Throughput:** Number of predictions per second, indicating model inference performance at scale

---

# Key Takeaways:
- GBT performance remained **consistent and stable across all dataset sizes**  
- High inference throughput proves suitability for real-time or batch deployment scenarios  
- Slight improvement in RMSE at smaller dataset subset likely due to reduced variance/noise

## Scalability & Optimization

To ensure performance and robustness at scale, the entire pipeline was deployed on **Google Cloud Platform (GCP)** using **Apache Spark clusters on Dataproc**.

---

### Cluster Tuning & Resource Configuration
- Dynamically scaled clusters based on dataset size (10% → 100%)
- Tuned Spark configurations including:
  - `spark.executor.memory`
  - `spark.executor.instances`
  - `spark.executor.cores`
- Enabled caching and persisted intermediate transformations where beneficial
- Used Parquet format to accelerate read/write operations and reduce I/O overhead

---

### Throughput Optimization
- Measured **inference throughput** across subsets: up to **33M+ predictions/sec**
- Maintained consistent model accuracy and execution time across all dataset sizes
- Pipeline architecture supports both **batch processing** and **future real-time scoring**

---

### Highlights:
- Demonstrates **scalable machine learning** deployment using Big Data tools
- Applies practical **Spark tuning techniques** to optimize cloud resource usage
- Validates end-to-end performance with real-world-sized data (82M+ records)

## How to Run Locally (Sample Simulation)

This project was built and executed on **Google Cloud Platform (GCP)** using **Dataproc Spark Clusters** and **Jupyter Notebooks**.  
The following example simulates a small-scale version for local testing and learning purposes.

---

### 1. Start a Local PySpark Session

> Skip this step if you're using GCP Dataproc Notebook — SparkSession is already active.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("flight_price_prediction") \
    .getOrCreate()
```

### 2. Load a Small Dataset Locally

Use Parquet if available, or fallback to CSV for basic testing.

```
# Load from a local Parquet file
df = spark.read.parquet("path/to/sample_data.parquet")

# Or load from a CSV file
df = spark.read.csv("path/to/sample_data.csv", header=True, inferSchema=True)
```

### 3. Apply Data Preprocessing & Feature Engineering

Transform your data using steps such as:

- Drop unnecessary columns (e.g., fareBasisCode, legId)
- Create `days_until_flight`, `is_peak_season`, etc.
- Use `StringIndexer` and `OneHotEncoder` for categorical fields
- Assemble features using VectorAssembler

```
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import datediff, to_date

# Example: Feature - Days Until Flight
df = df.withColumn("days_until_flight", datediff(to_date("flightDate"), to_date("bookingDate")))
```

### 4. Train ML Model (e.g., Gradient Boosted Trees)

Use Spark MLlib for training regression models.

```
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# Define your GBT model
gbt = GBTRegressor(featuresCol="features", labelCol="totalFare", maxIter=50)

# Build your pipeline
pipeline = Pipeline(stages=[
    # Add indexers, encoders, vectorAssembler here
    gbt
])

# Fit and predict
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

**Note**

- This local example is for simulation purposes only.
- The full pipeline was executed on GCP Dataproc with the complete 82M-row dataset stored in Google Cloud Storage (GCS).

## Skills Demonstrated

- **Big Data Engineering**: Efficient processing of 82 million+ records using Apache Spark and PySpark  
- **Distributed Computing**: Scalable data pipelines on Google Cloud Dataproc clusters  
- **Feature Engineering**: Creating meaningful features such as `days_until_flight`, `is_peak_season`, and encoding categorical variables for ML models  
- **Machine Learning**: Building and evaluating supervised regression models (Linear Regression, Gradient Boosted Trees) using Spark MLlib  
- **Performance Optimization**: Cluster tuning (executor memory, cores, instances) and throughput measurement across dataset sizes  
- **Cloud Architecture**: Leveraging Google Cloud Storage (GCS) for data management and Dataproc for cluster orchestration  
- **Data Storage & Formats**: Utilizing Parquet columnar format for efficient big data storage and querying  
- **Exploratory Data Analysis (EDA)**: Outlier detection, correlation analysis, and route popularity insights  
- **End-to-End Pipeline Design**: From raw data ingestion, cleaning, transformation, modeling, to scalable inference  

## Project Setup / Installation

### Option 1: Run on Google Cloud Platform (Recommended for full dataset)

1. **Create a Google Cloud account** and set up a project with billing enabled.  
2. **Enable the Dataproc and Cloud Storage APIs** in your GCP project.  
3. Upload your dataset (Parquet or CSV) to a **Google Cloud Storage (GCS) bucket**.  
4. Create a **Dataproc cluster** with suitable configuration (memory, CPUs).  
5. Use the **Jupyter notebook interface** on Dataproc to run the pipeline notebooks directly in the cloud environment.  
6. Make sure your PySpark environment and Spark MLlib libraries are available on the cluster.

### Option 2: Run Locally (for small-scale testing)

1. Install **Apache Spark** and **PySpark** on your local machine.  
2. Clone this repository and navigate to the project folder.  
3. Prepare a **small sample dataset** (CSV or Parquet) for local simulation.  
4. Run the local PySpark session and execute preprocessing, feature engineering, and model training as per the example in the **How to Run Locally** section.

---

### Prerequisites

- Python 3.7+  
- Java 8+ (required for Spark)  
- Apache Spark 3.x  
- PySpark  
- Google Cloud SDK (if working on GCP)  

---

### Helpful commands

```bash
# Install PySpark locally
pip install pyspark

# Start pyspark shell
pyspark
```

## Challenges & Learnings

- **Handling massive data volumes:** Efficiently processing 82 million rows required careful tuning of Spark cluster resources and optimizing data transformations to prevent bottlenecks.  
- **Data quality issues:** Missing values and inconsistent records demanded robust cleaning strategies, including group-based imputation and filtering low-frequency routes.  
- **Feature engineering complexity:** Creating meaningful features like `days_until_flight` and `is_peak_season` required domain knowledge and precise date handling.  
- **Balancing model complexity vs. scalability:** While Gradient Boosted Trees improved accuracy, training time and resource usage increased, necessitating careful trade-offs.  
- **Cloud infrastructure management:** Setting up and managing Dataproc clusters on GCP to scale compute resources dynamically was crucial for performance and cost efficiency.  
- **Distributed ML workflows:** Ensuring that model training and inference scales seamlessly across datasets of varying sizes involved thorough pipeline design and testing.

---

## Future Work

- **Enhance feature set:** Incorporate additional features such as competitor pricing, weather data, or customer booking behavior to improve prediction accuracy.  
- **Automate pipeline orchestration:** Use workflow managers like Apache Airflow to automate ETL, model training, and deployment steps.  
- **Model experimentation:** Explore advanced ML models such as XGBoost, LightGBM, or deep learning architectures for better performance.  
- **Real-time prediction service:** Develop a REST API or streaming pipeline for live flight price predictions.  
- **Cost optimization:** Implement dynamic cluster scaling and spot instances on GCP to reduce compute expenses.  
- **Comprehensive monitoring:** Add monitoring and logging to track pipeline health and model drift over time.

---

These enhancements would further mature the pipeline into a production-grade solution capable of supporting real-time, cost-efficient flight pricing intelligence.
