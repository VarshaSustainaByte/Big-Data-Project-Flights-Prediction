# Scalable Flight Price Prediction Using Big Data and Machine Learning

## Overview

This project builds a **scalable, production-ready data pipeline** to predict flight ticket prices using over **82 million** historical bookings from Expedia. It leverages **Apache Spark (PySpark)** and **Google Cloud Platform (GCP)** to demonstrate the full lifecycle of **Big Data Engineering and Machine Learning**, including:

- Ingesting and cleaning large-scale flight booking data
- Engineering features such as days until flight, seasonal indicators, and route popularity
- Storing optimized datasets in **Parquet format** on **Google Cloud Storage (GCS)**.
- Training and evaluating regression models (Linear Regression, Gradient Boosted Trees) using **Spark MLlib**.
- Scaling the entire workflow across varying dataset sizes using **Dataproc Spark clusters**.

--- 

## Real-World Applications & Impact

- Help **travelers** make cost-effective booking decisions by predicting fare trends
- Enable **airlines** to optimize pricing strategies using demand seasonality and booking behavior
- Empower **engineers**  and **data scientists** to build and scale predictive pipelines using real-world Big Data tools

The end result is a robust, cloud-deployed pipeline capable of handling and analyzing **massive datasets** with efficiency — demonstrating expertise in **data engineering, distributed processing, and applied machine learning**.

---

## Project Objectives

- Build a **scalable data pipeline** to process and analyze 82 million flight records using Apache Spark on Google Cloud.
- Engineer key features such as `days until flight`, `peak season`, `route popularity` to improve fare prediction accuracy.
- Train **supervised machine learning models** (Linear Regression, Gradient Boosted Trees) to forecast flight prices.
- Ensure **scalability** and performance optimization across various dataset sizes using **Dataproc Spark clusters**.
- Evaluate models using metrics like **RMSE, MAE, R²,** and benchmark **inference throughput** for production readiness.
- Deliver insights to help **travelers book smarter** and enable **airlines to analyze fare trends more effectively**.

---

## Dataset Overview

- **Source:** Expedia flight booking data from Kaggle  
- **Period:** April 2022 – October 2022  
- **Size:** Over **82 million rows** of historical flight records  
- **Storage Format:** Stored as **Parquet files** on **Google Cloud Storage (GCS)** for efficient quering and distributed access
- **Key Features Include:**
  - **Flight details:**
    - Departure and arrival airport codes
    - Travel distance and duration
    - Number of seats remaining
  - **Booking behavior:**
    - Booking date vs. flight date (used to compute `days_until_flight`)
    - Refundability and fare basis code
  - **Fare components:**
    - Base fare, total fare, and ticket class
  - **Categorical indicators:**
    - Flight type (non-stop, connecting)
    - Cabin class (economy, premium, etc.)

This dataset offers a robust foundation for modeling flight fare predictions using temporal, behavioral, and seasonal features.

---

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

These tools were selected for their ability to handle massive datasets efficiently, support distributed computation, and deliver production-ready machine learning pipelines in the cloud.

---

## Data Processing & Feature Engineering at Scale

- Removed irrelevant and duplicate columns such as `fareBasisCode` and `legId` to streamline data  
- Filtered routes to keep the top 50% by frequency, improving model focus and efficiency  
- Engineered key features including:  
  - `days_until_flight`: calculated from booking date and flight date to capture booking lead time  
  - `is_peak_season`: binary indicator for summer and Labor Day week to model seasonal demand  
  - `travelDurationMinutes`: total flight duration in minutes for fare prediction  
- Handled missing values by imputing group-based averages for continuous features  
- Encoded categorical variables (e.g., airport codes, flight type) using Spark’s `StringIndexer` and `OneHotEncoder` to prepare data for ML models  
- Stored processed data efficiently in **Parquet format** for fast access and downstream modeling

---

##  Exploratory Data Analysis (EDA)

- Explored route distribution to understand travel frequency across origin-destination pairs  
- Identified popular routes and high-traffic airports to inform feature engineering  
- Analyzed fare distributions to detect pricing trends, seasonality, and booking behavior  
- Performed outlier detection using **Interquartile Range (IQR)** method on fare-related columns  
- Calculated correlation between total fare, travel distance, and time until flight  
- Identified patterns between refundable and non-refundable ticket types  
- Discovered ~125 distinct routes and over **3.9 million** unique non-refundable flights

---

## Machine Learning Models

This project implements two supervised regression models using **PySpark MLlib**:

### 1. Linear Regression (Baseline)
- Interpretable model to establish baseline performance  
- Fast training time on large datasets  
- Validates feature relevance and data quality  

**Performance on Full Dataset:**  
- RMSE: ~127.4  
- MAE: ~81.6  
- R²: 0.553  

---

### 2. Gradient Boosted Trees (Optimized)
- Captures complex non-linear relationships between features and fare  
- Tuned for deeper trees and better generalization  
- Efficiently scales across dataset sizes from 10% to 100%  

**Why GBT?**  
- Robust to multicollinearity and missing values  
- Performs significantly better on structured/tabular data  
- Ideal for high-cardinality categorical features like airport codes  

**MLlib Configuration:**  
- `GBTRegressor(labelCol="totalFare", featuresCol="features", maxIter=50, maxDepth=5)`  
- Pipeline stages: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `GBTRegressor`  

All models were trained using **Spark ML pipelines**, ensuring modularity, scalability, and reproducibility.

---

## Model Evaluation

Both models were evaluated on prediction accuracy and scalability using standard regression metrics:

| Dataset Subset | RMSE     | MAE     | R²     | Inference Speed (preds/sec) |
|----------------|----------|---------|--------|------------------------------|
| 10%            | ~28.06   | ~81.8   | 0.549  | 3.1M+                        |
| 25%            | ~127.5   | ~81.9   | 0.550  | 17M+                         |
| 50%            | ~127.6   | ~81.8   | 0.552  | 33.9M+                       |
| 75%            | ~127.4   | ~81.7   | 0.553  | 29.8M+                       |
| 100%           | ~127.4   | ~81.6   | 0.553  | 31.8M+                       |

### Metrics Used:
- **Root Mean Squared Error (RMSE):** Measures prediction error magnitude  
- **Mean Absolute Error (MAE):** Captures average error per prediction  
- **R² Score:** Indicates how well the model explains variance in ticket prices  
- **Throughput:** Number of predictions per second, indicating model inference performance at scale

### Key Takeaways:
- GBT performance remained **consistent and stable across all dataset sizes**.
- High inference throughput demonstrates suitability for real-time or batch deployment.
- Slight RMSE improvement on smaller subsets likely reflects reduced variance/noise.

---

## Scalability & Optimization

The entire pipeline was deployed on **Google Cloud Platform (GCP)** using **Apache Spark** clusters on Dataproc to ensure performance and robustness at scale.

### Cluster Tuning & Resource Configuration
- Dynamically scaled clusters based on dataset size (10% → 100%)
- Tuned Spark configurations including:
  - `spark.executor.memory`
  - `spark.executor.instances`
  - `spark.executor.cores`
- Enabled caching and persisted intermediate transformations where beneficial
- Used Parquet format to accelerate read/write operations and reduce I/O overhead

### Throughput Optimization
- Measured **inference throughput** across dataset subsets, reaching up to **33M+ predictions/sec**
- Maintained consistent model accuracy and execution time across all dataset sizes
- Pipeline architecture supports both **batch processing** and **future real-time scoring**

---

## Project Setup & Execution

### Option 1: Run on Google Cloud Platform (Recommended for full dataset)

1. **Create a Google Cloud account** and set up a project with billing enabled.  
2. **Enable the Dataproc and Cloud Storage APIs** in your GCP project.  
3. Upload your dataset (Parquet or CSV) to a **Google Cloud Storage (GCS) bucket**.  
4. Create a **Dataproc cluster** with suitable configuration (memory, CPUs).  
5. Use the **Jupyter notebook interface** on Dataproc to run the pipeline notebooks directly in the cloud environment.  
6. Make sure your PySpark environment and Spark MLlib libraries are available on the cluster.

### Option 2: Run Locally (for small-scale testing)

This section provides a simplified local setup to test the pipeline with a small sample dataset.  
The full pipeline is designed for Google Cloud Dataproc with 82M+ records.

#### 1. Start a local PySpark session:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("flight_price_prediction").getOrCreate()
```

#### 2. Load a Small Dataset Locally

```
df = spark.read.parquet("path/to/sample_data.parquet")  # or CSV with header and schema inference
```

#### 3. Apply Data Preprocessing & Feature Engineering

- Drop unnecessary columns
- Create features like days_until_flight
- Encode categorical variables

#### 4. Train ML Model (e.g., Gradient Boosted Trees)

```
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# Define pipeline stages and train model
```

### **Note**

- This local example is for simulation purposes only.
- The full pipeline was executed on GCP Dataproc with the complete 82M-row dataset stored in Google Cloud Storage (GCS).

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
---

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
- **Data Visualization:** Created charts and dashboards during EDA to communicate insights on fare trends and route popularity.

---

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

These enhancements would further mature the pipeline into a production-grade solution capable of supporting real-time, cost-efficient flight pricing intelligence.

---

## Credits

This project was collaboratively developed by:

- [**Aradhana Ramamoorthy**]
- [**Zhu Hsuan Lin**]  
- [**Varsha Abraham**]

---

