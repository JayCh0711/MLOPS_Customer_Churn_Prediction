"""
Model training module with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn
import joblib
import logging
import os
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training class with MLflow experiment tracking
    """
    
    def __init__(self, experiment_name: str = "customer-churn-prediction"):
        """
        Initialize trainer with MLflow experiment
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.model = None
        self.metrics = {}
        self.best_model = None
        self.best_metrics = {}
        
        # Set up MLflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow experiment: {experiment_name}")
    
    def get_model(self, algorithm: str, params: Dict[str, Any] = None):
        """
        Get model instance based on algorithm name
        
        Args:
            algorithm: Name of algorithm
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        params = params or {}
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                min_samples_split=params.get('min_samples_split', 5),
                random_state=params.get('random_state', 42)
            ),
            'logistic_regression': LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
        }
        
        if algorithm not in models:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(models.keys())}")
        
        return models[algorithm]
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        algorithm: str,
        params: Dict[str, Any] = None,
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Train model with MLflow tracking
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            algorithm: Algorithm name
            params: Model parameters
            run_name: Optional run name
            
        Returns:
            Dictionary with training results
        """
        params = params or {}
        run_name = run_name or f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Training {algorithm} model...")
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log dataset info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Get and train model
            self.model = self.get_model(algorithm, params)
            
            logger.info("Fitting model...")
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_train_prob = self.model.predict_proba(X_train)[:, 1]
                y_test_prob = self.model.predict_proba(X_test)[:, 1]
            else:
                y_train_prob = None
                y_test_prob = None
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_prob)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            # Log metrics
            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)
            
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            
            # Log confusion matrix as artifact
            cm = confusion_matrix(y_test, y_test_pred)
            cm_dict = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
            
            # Save confusion matrix
            os.makedirs("models/metrics", exist_ok=True)
            cm_path = "models/metrics/confusion_matrix.json"
            with open(cm_path, 'w') as f:
                json.dump(cm_dict, f, indent=2)
            mlflow.log_artifact(cm_path)
            
            # Log classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)
            report_path = "models/metrics/classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = "models/metrics/feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"churn-model-{algorithm}"
            )
            
            # Store results
            self.metrics = {
                'train': train_metrics,
                'test': test_metrics,
                'confusion_matrix': cm_dict
            }
            
            results = {
                'run_id': run.info.run_id,
                'algorithm': algorithm,
                'params': params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'confusion_matrix': cm_dict
            }
            
            logger.info(f"Training complete. Run ID: {run.info.run_id}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
            logger.info(f"Test ROC AUC: {test_metrics.get('roc_auc', 'N/A')}")
            
            return results
    
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        algorithms: list = None,
        params_dict: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models and compare
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            algorithms: List of algorithms to train
            params_dict: Dictionary of parameters for each algorithm
            
        Returns:
            Dictionary with all results and best model info
        """
        algorithms = algorithms or ['random_forest', 'gradient_boosting', 'logistic_regression']
        params_dict = params_dict or {}
        
        all_results = {}
        best_f1 = 0
        best_algorithm = None
        
        for algorithm in algorithms:
            params = params_dict.get(algorithm, {})
            
            try:
                results = self.train(
                    X_train, y_train, X_test, y_test,
                    algorithm=algorithm,
                    params=params
                )
                all_results[algorithm] = results
                
                # Track best model based on F1 score
                if results['test_metrics']['f1_score'] > best_f1:
                    best_f1 = results['test_metrics']['f1_score']
                    best_algorithm = algorithm
                    self.best_model = self.model
                    self.best_metrics = results['test_metrics']
                    
            except Exception as e:
                logger.error(f"Error training {algorithm}: {e}")
                all_results[algorithm] = {'error': str(e)}
        
        # Summary
        summary = {
            'all_results': all_results,
            'best_algorithm': best_algorithm,
            'best_f1_score': best_f1,
            'best_metrics': self.best_metrics
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Best Model: {best_algorithm}")
        logger.info(f"Best F1 Score: {best_f1:.4f}")
        logger.info(f"{'='*50}")
        
        return summary
    
    def save_model(self, path: str):
        """
        Save the best model to disk
        
        Args:
            path: Path to save model
        """
        if self.best_model is None:
            self.best_model = self.model
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        logger.info(f"Model saved to {path}")


def load_data(data_dir: str = "data/processed") -> Tuple:
    """Load train and test data"""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    
    return X_train, X_test, y_train, y_test


def load_params(params_path: str = "params.yaml") -> Dict:
    """Load parameters from YAML file"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def train_model():
    """Main training function"""
    # Load data
    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Load parameters
    params = load_params()
    model_params = params.get('model', {})
    
    # Initialize trainer
    trainer = ModelTrainer(experiment_name="customer-churn-prediction")
    
    # Define algorithms and their parameters
    algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression']
    
    params_dict = {
        'random_forest': {
            'n_estimators': model_params.get('n_estimators', 100),
            'max_depth': model_params.get('max_depth', 10),
            'min_samples_split': model_params.get('min_samples_split', 5),
            'min_samples_leaf': model_params.get('min_samples_leaf', 2),
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
    }
    
    # Train multiple models
    summary = trainer.train_multiple_models(
        X_train, y_train, X_test, y_test,
        algorithms=algorithms,
        params_dict=params_dict
    )
    
    # Save best model
    trainer.save_model("models/best_model.joblib")
    
    # Save training summary
    os.makedirs("models/metrics", exist_ok=True)
    summary_path = "models/metrics/training_summary.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    summary_serializable = convert_to_serializable(summary)
    
    with open(summary_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_path}")
    
    return summary


if __name__ == "__main__":
    train_model()