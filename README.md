# **🛒 Store Sales \- Time Series Forecasting**

## **📋 Overview**

This project is an end-to-end solution for the **Store Sales \- Time Series Forecasting** Kaggle competition. The goal is to predict grocery sales for Corporación Favorita, a large Ecuadorian grocery retailer.

Beyond standard modeling, this repository implements a **production-ready architecture** including:

* A robust **Preprocessing Pipeline** handling holidays, oil prices, and transactions.  
* **Facebook Prophet** modeling with hyperparameter tuning.  
* An interactive **Streamlit Web Application** for real-time forecasting and scenario analysis.  
* A dedicated **Monitoring System** to track model drift (MAE/RMSE) and trigger retraining alerts.

## **✨ Key Features**

### **1\. Advanced Data Pipeline (preprocessing.py)**

* Automated cleaning of auxiliary datasets (Oil, Holidays, Transactions).  
* Feature Engineering: Extraction of seasonality (Year, Month, Day of Week).  
* Handling of "Bridge", "Transfer", and "Event" holiday types with priority logic.  
* Encoding and Scaling utilizing Scikit-Learn pipelines.

### **2\. Exploratory Data Analysis (EDA.py)**

* Automated generation of descriptive statistics.  
* Visualizations for:  
  * Sales distribution by state and store type.  
  * Impact of promotions on sales.  
  * Seasonal trends and correlation heatmaps.

### **3\. Forecasting Model (best\_model.py)**

* Utilization of **Facebook Prophet** for time-series forecasting.  
* Configured for additive seasonality and custom holiday effects.  
* Includes logic for grid search and hyperparameter optimization.

### **4\. Model Monitoring (Monitoring.py)**

* A custom ProphetModelMonitor class.  
* Detects **Concept Drift** by comparing live metrics against baseline MAE/RMSE.  
* Provides severity alerts (Healthy, Warning, Drift Detected) and retraining recommendations.

### **5\. Interactive Dashboards**

* **Streamlit App (streamlit.py):** The main user interface allowing users to toggle between City Sales Dashboards and Forecast modes.  
* **Analytics (DashBoard.py):** Detailed Dash/Plotly visualizations for deep-dive analytics.

## **📂 Project Structure**

├── data/                   \# Raw CSV files (train, test, holidays, etc.)  
├── models/                 \# Serialized models (.pkl files)  
│   ├── prophet\_tuned\_model.pkl  
│   └── ...  
├──  preprocessing.py    \# Data cleaning and transformation pipeline  
├──  EDA.py              \# Exploratory Data Analysis scripts  
├──  best\_model.py       \# Prophet model training and configuration  
├──  Monitoring.py       \# Drift detection and model health checks  
├──  DashBoard.py        \# Plotly Dash visualization components  
├── streamlit.py            \# Main entry point for the Web App  
├── requirements.txt        \# Python dependencies  
└── README.md               \# Project documentation

## **🚀 Installation & Setup**

### **Prerequisites**

* Python 3.8 or higher  
* pip package manager

### **Step 1: Clone the Repository**

git clone \[[https://github.com/your-username/store-sales-forecasting.git\](https://github.com/your-username/store-sales-forecasting.git](https://github.com/Amr2272/Final-Project.git))  
cd store-sales-forecasting

### **Step 2: Install Dependencies**

pip install \-r requirements.txt

### **Step 3: Data Setup**

Download the dataset from [Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) and place the CSV files (or Data.zip) in the root directory or a data/ folder as referenced in preprocessing.py.

## **🖥️ Usage**

### **Running the Streamlit App**

To launch the main forecasting interface:

streamlit run streamlit.py

*The app will open in your browser at http://localhost:8501.*

### **Running the Analysis Dashboard**

To view the detailed analytics dashboard:

python DashBoard.py

### **Training the Model**

To retrain the model using the pipeline:

from best\_model import load\_and\_preprocess, fit\_prophet\_model

\# Load data  
train, test \= load\_and\_preprocess()

\# Fit model  
model \= fit\_prophet\_model(train)

## **📊 Model Monitoring Logic**

The project includes a drift detection mechanism implemented in Monitoring.py. It works as follows:

1. **Baseline Establishment:** Saves initial training MAE and RMSE.  
2. **Live Logging:** Records new predictions vs. actuals.  
3. **Drift Calculation:**  
   if current\_metric \> baseline \* (1 \+ threshold):  
       status \= "Drift Detected"

4. **Feedback:** The system recommends retraining if severity is high.

## **🛠️ Tech Stack**

* **Core:** Python, Pandas, NumPy  
* **ML & Stats:** Prophet, Scikit-Learn, Statsmodels, SciPy  
* **Visualization:** Plotly, Seaborn, Matplotlib  
* **Web Frameworks:** Streamlit, Dash  
* **Boosting:** XGBoost, LightGBM (included in requirements for experimental comparisons)

## **📄 License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## **🤝 Contributing**

Contributions are welcome\! Please open an issue or submit a pull request for any improvements.

1. Fork the Project  
2. Create your Feature Branch (git checkout \-b feature/AmazingFeature)  
3. Commit your Changes (git commit \-m 'Add some AmazingFeature')  
4. Push to the Branch (git push origin feature/AmazingFeature)  
5. Open a Pull Request
