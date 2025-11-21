import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ProphetModelMonitor:
    """
    Comprehensive monitoring system for Prophet forecasting models.
    Tracks performance, detects drift, and provides feedback for continuous improvement.
    """
    
    def __init__(self, baseline_mae=None, baseline_rmse=None, drift_threshold=0.15):
        """
        Initialize the monitoring system.
        
        Parameters:
        -----------
        baseline_mae : float
            Baseline Mean Absolute Error from initial model training
        baseline_rmse : float
            Baseline Root Mean Squared Error from initial model training
        drift_threshold : float
            Threshold for detecting significant performance drift (default: 15%)
        """
        self.baseline_mae = baseline_mae
        self.baseline_rmse = baseline_rmse
        self.drift_threshold = drift_threshold
        self.performance_log = []
        self.drift_alerts = []
        self.prediction_errors = []
        
    def log_prediction(self, date, actual, predicted, features=None):
        """
        Log a single prediction with actual value for monitoring.
        
        Parameters:
        -----------
        date : datetime or str
            Date of prediction
        actual : float
            Actual observed value
        predicted : float
            Model predicted value
        features : dict
            Optional dictionary of feature values for drift detection
        """
        error = actual - predicted
        abs_error = abs(error)
        pct_error = (abs_error / actual * 100) if actual != 0 else 0
        
        log_entry = {
            'date': pd.to_datetime(date),
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'abs_error': abs_error,
            'pct_error': pct_error,
            'features': features or {}
        }
        
        self.prediction_errors.append(log_entry)
        
    def calculate_metrics(self, window='7D'):
        """
        Calculate performance metrics over a rolling window.
        
        Parameters:
        -----------
        window : str
            Time window for metric calculation (e.g., '7D', '30D')
        
        Returns:
        --------
        pd.DataFrame : Performance metrics over time
        """
        if not self.prediction_errors:
            print("No predictions logged yet.")
            return None
            
        df = pd.DataFrame(self.prediction_errors)
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # Calculate rolling metrics
        metrics_df = pd.DataFrame({
            'mae': df['abs_error'].rolling(window).mean(),
            'rmse': df['error'].rolling(window).apply(lambda x: np.sqrt((x**2).mean())),
            'mape': df['pct_error'].rolling(window).mean(),
            'bias': df['error'].rolling(window).mean(),
            'count': df['abs_error'].rolling(window).count()
        })
        
        return metrics_df
    
    def detect_drift(self, current_mae, current_rmse):
        """
        Detect if model performance has drifted significantly.
        
        Parameters:
        -----------
        current_mae : float
            Current Mean Absolute Error
        current_rmse : float
            Current Root Mean Squared Error
        
        Returns:
        --------
        dict : Drift detection results
        """
        drift_result = {
            'drift_detected': False,
            'mae_drift_pct': 0,
            'rmse_drift_pct': 0,
            'severity': 'none',
            'timestamp': datetime.now()
        }
        
        if self.baseline_mae and self.baseline_rmse:
            mae_drift = (current_mae - self.baseline_mae) / self.baseline_mae
            rmse_drift = (current_rmse - self.baseline_rmse) / self.baseline_rmse
            
            drift_result['mae_drift_pct'] = mae_drift * 100
            drift_result['rmse_drift_pct'] = rmse_drift * 100
            
            # Check if drift exceeds threshold
            if abs(mae_drift) > self.drift_threshold or abs(rmse_drift) > self.drift_threshold:
                drift_result['drift_detected'] = True
                
                # Determine severity
                max_drift = max(abs(mae_drift), abs(rmse_drift))
                if max_drift > 0.3:
                    drift_result['severity'] = 'critical'
                elif max_drift > 0.2:
                    drift_result['severity'] = 'high'
                else:
                    drift_result['severity'] = 'moderate'
                
                # Log alert
                alert = {
                    'timestamp': datetime.now(),
                    'mae_drift': mae_drift * 100,
                    'rmse_drift': rmse_drift * 100,
                    'severity': drift_result['severity']
                }
                self.drift_alerts.append(alert)
                
        return drift_result
    
    def feature_distribution_shift(self, baseline_features, current_features, feature_name):
        """
        Detect distribution shift in input features using Kolmogorov-Smirnov test.
        
        Parameters:
        -----------
        baseline_features : array-like
            Feature values from baseline period
        current_features : array-like
            Current feature values
        feature_name : str
            Name of the feature being tested
        
        Returns:
        --------
        dict : Distribution shift test results
        """
        # Perform KS test
        statistic, p_value = stats.ks_2samp(baseline_features, current_features)
        
        result = {
            'feature': feature_name,
            'ks_statistic': statistic,
            'p_value': p_value,
            'shift_detected': p_value < 0.05,
            'baseline_mean': np.mean(baseline_features),
            'current_mean': np.mean(current_features),
            'baseline_std': np.std(baseline_features),
            'current_std': np.std(current_features)
        }
        
        return result
    
    def generate_monitoring_report(self, save_path='monitoring_report.html'):
        """
        Generate a comprehensive monitoring report with visualizations.
        """
        if not self.prediction_errors:
            print("No predictions logged. Cannot generate report.")
            return
        
        df = pd.DataFrame(self.prediction_errors)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Prophet Model Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].plot(df['date'], df['actual'], label='Actual', marker='o', markersize=3)
        axes[0, 0].plot(df['date'], df['predicted'], label='Predicted', marker='o', markersize=3)
        axes[0, 0].set_title('Actual vs Predicted Sales')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error Distribution
        axes[0, 1].hist(df['error'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(df['error'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {df["error"].mean():.2f}')
        axes[0, 1].set_title('Prediction Error Distribution')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling MAE
        df_indexed = df.set_index('date')
        rolling_mae = df_indexed['abs_error'].rolling('7D').mean()
        axes[1, 0].plot(rolling_mae.index, rolling_mae.values, color='orange', linewidth=2)
        if self.baseline_mae:
            axes[1, 0].axhline(self.baseline_mae, color='green', linestyle='--', 
                              label=f'Baseline MAE: {self.baseline_mae:.2f}')
            axes[1, 0].axhline(self.baseline_mae * (1 + self.drift_threshold), 
                              color='red', linestyle='--', label='Drift Threshold')
        axes[1, 0].set_title('Rolling MAE (7-day window)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage Error Over Time
        axes[1, 1].scatter(df['date'], df['pct_error'], alpha=0.5, s=20)
        axes[1, 1].axhline(df['pct_error'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["pct_error"].mean():.2f}%')
        axes[1, 1].set_title('Percentage Error Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Percentage Error (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Residuals vs Predicted (Homoscedasticity check)
        axes[2, 0].scatter(df['predicted'], df['error'], alpha=0.5, s=20)
        axes[2, 0].axhline(0, color='red', linestyle='--')
        axes[2, 0].set_title('Residuals vs Predicted Values')
        axes[2, 0].set_xlabel('Predicted Sales')
        axes[2, 0].set_ylabel('Residuals')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        metrics_text = f"""
        Performance Metrics (Overall):
        
        MAE: {df['abs_error'].mean():.2f}
        RMSE: {np.sqrt((df['error']**2).mean()):.2f}
        MAPE: {df['pct_error'].mean():.2f}%
        
        Bias: {df['error'].mean():.2f}
        Max Error: {df['abs_error'].max():.2f}
        
        Total Predictions: {len(df)}
        """
        
        if self.baseline_mae:
            current_mae = df['abs_error'].mean()
            drift_pct = ((current_mae - self.baseline_mae) / self.baseline_mae) * 100
            metrics_text += f"\nDrift from Baseline: {drift_pct:.2f}%"
        
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                       verticalalignment='center', family='monospace')
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('monitoring_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Monitoring dashboard saved as 'monitoring_dashboard.png'")
    
    def check_model_health(self):
        """
        Perform comprehensive health check on the model.
        
        Returns:
        --------
        dict : Health check results with recommendations
        """
        if not self.prediction_errors:
            return {"status": "insufficient_data", "message": "Not enough predictions logged"}
        
        df = pd.DataFrame(self.prediction_errors)
        current_mae = df['abs_error'].mean()
        current_rmse = np.sqrt((df['error']**2).mean())
        
        health_report = {
            'timestamp': datetime.now(),
            'current_mae': current_mae,
            'current_rmse': current_rmse,
            'num_predictions': len(df),
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check for drift
        if self.baseline_mae and self.baseline_rmse:
            drift = self.detect_drift(current_mae, current_rmse)
            if drift['drift_detected']:
                health_report['status'] = 'degraded'
                health_report['issues'].append(f"Performance drift detected ({drift['severity']})")
                health_report['recommendations'].append("Consider retraining the model with recent data")
        
        # Check for bias
        mean_error = df['error'].mean()
        if abs(mean_error) > current_mae * 0.5:
            health_report['issues'].append(f"Significant bias detected: {mean_error:.2f}")
            health_report['recommendations'].append("Model systematically over/under-predicts")
        
        # Check for increasing variance
        recent_errors = df.tail(50)['abs_error']
        older_errors = df.head(50)['abs_error']
        if len(recent_errors) > 30 and len(older_errors) > 30:
            if recent_errors.std() > older_errors.std() * 1.5:
                health_report['issues'].append("Increasing error variance")
                health_report['recommendations'].append("Check for changes in data patterns or external factors")
        
        if not health_report['issues']:
            health_report['recommendations'].append("Model is performing well. Continue monitoring.")
        
        return health_report
    
    def save_monitoring_data(self, filepath='monitoring_log.csv'):
        """Save monitoring data to CSV for long-term tracking."""
        if not self.prediction_errors:
            print("No data to save.")
            return
        
        df = pd.DataFrame(self.prediction_errors)
        df.to_csv(filepath, index=False)
        print(f"✓ Monitoring data saved to {filepath}")
    
    def load_monitoring_data(self, filepath='monitoring_log.csv'):
        """Load monitoring data from CSV."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            self.prediction_errors = df.to_dict('records')
            print(f"✓ Loaded {len(self.prediction_errors)} prediction records")
        except FileNotFoundError:
            print(f"File {filepath} not found.")


# Example Usage and Integration
def integrate_monitoring_with_prophet(train, test, model, forecast):
    """
    Integrate monitoring system with your existing Prophet model.
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data with 'ds' and 'y' columns
    test : pd.DataFrame
        Test data with 'ds' and 'y' columns
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast output from Prophet
    """
    
    # Calculate baseline metrics from test set
    y_true = test['y'].values
    y_pred = forecast['yhat'].iloc[-len(test):].values
    
    baseline_mae = mean_absolute_error(y_true, y_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    
    # Initialize monitoring system
    monitor = ProphetModelMonitor(
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        drift_threshold=0.15
    )
    
    # Log test predictions
    for i, (idx, row) in enumerate(test.iterrows()):
        actual = row['y']
        predicted = y_pred[i]
        
        monitor.log_prediction(
            date=row['ds'],
            actual=actual,
            predicted=predicted
        )
    
    # Generate monitoring report
    monitor.generate_monitoring_report()
    
    # Check model health
    health = monitor.check_model_health()
    print("\n" + "="*60)
    print("MODEL HEALTH CHECK")
    print("="*60)
    print(f"Status: {health['status'].upper()}")
    print(f"Current MAE: {health['current_mae']:.2f}")
    print(f"Current RMSE: {health['current_rmse']:.2f}")
    
    if health['issues']:
        print("\nIssues Detected:")
        for issue in health['issues']:
            print(f"  ⚠ {issue}")
    
    if health['recommendations']:
        print("\nRecommendations:")
        for rec in health['recommendations']:
            print(f"  → {rec}")
    
    # Save monitoring data
    monitor.save_monitoring_data()
    
    return monitor


# Continuous Feedback Loop Implementation
def continuous_monitoring_workflow(monitor, new_actual, new_predicted, new_date):
    """
    Workflow for continuous monitoring in production.
    Call this function each time you make a new prediction.
    
    Parameters:
    -----------
    monitor : ProphetModelMonitor
        Existing monitor instance
    new_actual : float
        New actual observed value
    new_predicted : float
        New predicted value
    new_date : datetime
        Date of prediction
    """
    
    # Log new prediction
    monitor.log_prediction(new_date, new_actual, new_predicted)
    
    # Calculate current metrics
    df = pd.DataFrame(monitor.prediction_errors)
    recent_predictions = df.tail(30)  # Last 30 predictions
    
    if len(recent_predictions) >= 30:
        current_mae = recent_predictions['abs_error'].mean()
        current_rmse = np.sqrt((recent_predictions['error']**2).mean())
        
        # Check for drift
        drift = monitor.detect_drift(current_mae, current_rmse)
        
        if drift['drift_detected']:
            print(f"\n🚨 ALERT: Model drift detected!")
            print(f"   Severity: {drift['severity']}")
            print(f"   MAE Drift: {drift['mae_drift_pct']:.2f}%")
            print(f"   RMSE Drift: {drift['rmse_drift_pct']:.2f}%")
            print(f"   → Recommendation: Retrain model with recent data")
            
            return 'retrain_needed'
    
    return 'healthy'


print("✓ ProphetModelMonitor class loaded successfully!")
print("\nUsage:")
print("1. monitor = integrate_monitoring_with_prophet(train, test, model, forecast)")
print("2. status = continuous_monitoring_workflow(monitor, actual, predicted, date)")
print("3. monitor.generate_monitoring_report()")
print("4. health = monitor.check_model_health()")
