# **🛒 Store Sales - Time Series Forecasting**

## **📋 Overview**

This project is an end-to-end solution for the **Store Sales - Time Series Forecasting** Kaggle competition. The goal is to predict grocery sales for Corporación Favorita, a large Ecuadorian grocery retailer.

Beyond standard modeling, this repository implements a **production-ready MLOps architecture** including:

* A robust **Preprocessing Pipeline** handling holidays, oil prices, and transactions.
* **Multi-model approach** (Prophet, ARIMA, LightGBM) with automated selection.
* **MLOps infrastructure** using DVC, MLflow, and Git for versioning and experiment tracking.
* An interactive **Streamlit Web Application** for real-time forecasting and scenario analysis.
* A dedicated **Monitoring System** to track model drift (MAE/RMSE) and trigger retraining alerts.

## **✨ Key Features**

### **1. Advanced Data Pipeline (preprocessing.py)**

* Automated cleaning of auxiliary datasets (Oil, Holidays, Transactions).
* Feature Engineering: Extraction of seasonality (Year, Month, Day of Week).
* Handling of "Bridge", "Transfer", and "Event" holiday types with priority logic.
* Encoding and Scaling utilizing Scikit-Learn pipelines.

### **2. Exploratory Data Analysis (EDA.py)**

* Automated generation of descriptive statistics.
* Visualizations for:
  * Sales distribution by state and store type.
  * Impact of promotions on sales.
  * Seasonal trends and correlation heatmaps.

### **3. Multi-Model Forecasting System**

* **Three competing models:** Facebook Prophet, ARIMA, and LightGBM.
* Automated hyperparameter tuning for each model.
* Systematic comparison using RMSE and MAE metrics.
* Best model automatically selected and promoted to production.

### **4. MLOps Infrastructure**

#### **Data & Model Versioning (DVC + Git)**
* **DVC (Data Version Control)** tracks large datasets and model binaries without bloating Git.
* Remote storage hosted on **DagsHub**.
* Three-way linkage ensures reproducibility: Code (Git) → Data (DVC) → Model (MLflow).

#### **Experiment Tracking (MLflow)**
* Centralized tracking of all training experiments via **DagsHub MLflow server**.
* Three dedicated experiments: `prophet_test`, `arima_test`, `lightgbm_test`.
* Logs parameters, metrics (RMSE, MAE), and model artifacts.
* Special `best_models` experiment aggregates top performers.

#### **Model Registry & Lifecycle**
* **MLflow Model Registry** manages model stages: None → Staging → Production → Archived.
* Automated model promotion based on performance metrics.
* Quick rollback capability using archived versions.

#### **End-to-End Pipeline**
1. **Data Retrieval:** Automated `dvc pull` from remote storage.
2. **Multi-Model Training:** Parallel training with hyperparameter tuning.
3. **Model Evaluation:** Best model selection based on lowest RMSE.
4. **Versioning:** Models tracked with DVC and pushed to remote storage.
5. **Deployment:** Stage transitions trigger deployment pipelines.
6. **Monitoring:** Continuous tracking via MLflow UI and monitoring system.

### **5. Model Monitoring (Monitoring.py)**

* A custom ProphetModelMonitor class.
* Detects **Concept Drift** by comparing live metrics against baseline MAE/RMSE.
* Provides severity alerts (Healthy, Warning, Drift Detected) and retraining recommendations.

### **6. Interactive Dashboards**

* **Streamlit App (streamlit.py):** The main user interface allowing users to toggle between City Sales Dashboards and Forecast modes.
* **Analytics (DashBoard.py):** Detailed Dash/Plotly visualizations for deep-dive analytics.

## **📂 Project Structure**

```
.
├── data/                    # Raw input files (train, test, holidays, etc.)
├── models/                  # Serialized models (.pkl files)
├── mlflow/
│   ├── mlflow.rar/          
├── src/                     # Contains all core Python source code
│   ├── preprocessing.py    # Data cleaning and transformation pipeline
│   ├── EDA.py              # Exploratory Data Analysis scripts
│   ├── best_model.py       # Prophet model training and configuration
│   ├── Monitoring.py       # Drift detection and model health checks
│   ├── DashBoard.py        # Plotly Dash visualization components
│   └── streamlit.py        # Main entry point for the Streamlit Web App
├── logs/
│   └── monitoring_log.csv  # Logs tracking model performance and drift
├── reports/
│   ├── cleaned Dataset and Analysis Report/
│   ├── Data Exploration Report/
│   ├── Forecasting Model Performance Report/
│   ├── Monitoring Setup Report/
│   ├── MLOps Report/       # MLOps infrastructure documentation
│   └── Final Report/
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## **🚀 Installation & Setup**

### **Prerequisites**

* Python 3.8 or higher
* pip package manager
* Git and DVC installed

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Amr2272/Final-Project.git
cd store-sales-forecasting
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3: Data Setup**

**Option A: Pull from DVC Remote**
```bash
dvc pull
```

**Option B: Manual Download**
Download the dataset from [Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) and place files in the data/ folder.

### **Step 4: MLflow Setup**

Start the MLflow tracking server:
```bash
mlflow ui --backend-store-uri sqlite:///D:/Final_Project/mlflow_project/mlflow.db
```
Access at: http://127.0.0.1:5000

## **🖥️ Usage**

### **Running the Streamlit App**

To launch the main forecasting interface:

```bash
streamlit run streamlit.py
```

*The app will open in your browser at http://localhost:8501.*

### **Running the Analysis Dashboard**

To view the detailed analytics dashboard:

```bash
python DashBoard.py
```

### **Training Models**

To retrain all models and track experiments:

```python
from best_model import train_all_models

# Train Prophet, ARIMA, and LightGBM
train_all_models()
```

### **Versioning Models**

After training, version your models with DVC:

```bash
dvc add models/
git add models.dvc
git commit -m "Update models"
dvc push
```

## **🔬 MLOps Workflow**

### **Experiment Tracking**

Each model type has its dedicated MLflow experiment:

| Experiment | Parameters Logged | Models |
|------------|-------------------|--------|
| `prophet_test` | changepoint_prior_scale, seasonality_mode, etc. | Prophet |
| `arima_test` | p, d, q | ARIMA |
| `lightgbm_test` | num_leaves, learning_rate, max_depth, etc. | LightGBM |

All experiments log:
* Training and test RMSE/MAE
* Model hyperparameters
* Model artifacts (.pkl files)

### **Model Selection & Promotion**

1. Models are evaluated in the `best_models` experiment
2. Best performer (lowest RMSE) is identified automatically
3. Top model is registered in MLflow Model Registry
4. Model transitions: Staging → Production
5. Deployment pipeline triggered on production promotion

### **Reproducibility**

Every production model is fully reproducible through:
* **Git commit hash** → Code version
* **DVC pointer file** → Data version
* **MLflow run ID** → Model artifact and metrics

## **📊 Model Monitoring Logic**

The project includes a drift detection mechanism implemented in Monitoring.py:

1. **Baseline Establishment:** Saves initial training MAE and RMSE.
2. **Live Logging:** Records new predictions vs. actuals.
3. **Drift Calculation:**
   ```python
   if current_metric > baseline * (1 + threshold):
       status = "Drift Detected"
   ```
4. **Feedback:** The system recommends retraining if severity is high.

## **🛠️ Tech Stack**

### **MLOps & Infrastructure**
* **Version Control:** Git, DVC (DagsHub)
* **Experiment Tracking:** MLflow
* **Model Registry:** MLflow Model Registry

### **Machine Learning**
* **Time Series:** Prophet, ARIMA (Statsmodels)
* **Gradient Boosting:** LightGBM, XGBoost
* **ML Utilities:** Scikit-Learn, SciPy

### **Data & Visualization**
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib

### **Web Frameworks**
* **Frontend:** Streamlit, Dash



## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

## **🤝 Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

