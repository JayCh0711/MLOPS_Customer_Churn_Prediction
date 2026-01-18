"""
Model evaluation module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import joblib
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation class for generating evaluation reports and plots
    """
    
    def __init__(self, model_path: str = "models/best_model.joblib"):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model
        """
        self.model = joblib.load(model_path)
        self.output_dir = "models/evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Loaded model from {model_path}")
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        if y_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc
        }
        
        return results
    
    def plot_confusion_matrix(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True
    ):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test features
            y_test: Test target
            save: Whether to save the plot
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save:
            path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {path}")
        
        plt.close()
    
    def plot_roc_curve(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True
    ):
        """
        Plot ROC curve
        
        Args:
            X_test: Test features
            y_test: Test target
            save: Whether to save the plot
        """
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model doesn't support probability prediction. Skipping ROC curve.")
            return
        
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save:
            path = os.path.join(self.output_dir, 'roc_curve.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            X_test: Test features
            y_test: Test target
            save: Whether to save the plot
        """
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model doesn't support probability prediction. Skipping PR curve.")
            return
        
        y_prob = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save:
            path = os.path.join(self.output_dir, 'precision_recall_curve.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {path}")
        
        plt.close()
    
    def plot_feature_importance(
        self, 
        feature_names: list,
        top_n: int = 15,
        save: bool = True
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        if not hasattr(self.model, 'feature_importances_'):
            # Create a placeholder plot for models without feature importances
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
            plt.title('Feature Importances')
            plt.axis('off')
            logger.warning("Model doesn't have feature importances. Creating placeholder plot.")
        else:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True).tail(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
        
        if save:
            path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {path}")
        
        plt.close()
    
    def generate_report(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> dict:
        """
        Generate complete evaluation report
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with complete evaluation
        """
        logger.info("Generating evaluation report...")
        
        # Get evaluation metrics
        results = self.evaluate(X_test, y_test)
        
        # Generate plots
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_roc_curve(X_test, y_test)
        self.plot_precision_recall_curve(X_test, y_test)
        self.plot_feature_importance(list(X_test.columns))
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return results


def evaluate_model():
    """Main evaluation function"""
    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    # Initialize evaluator
    evaluator = ModelEvaluator("models/best_model.joblib")
    
    # Generate report
    results = evaluator.generate_report(X_test, y_test)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"\nROC AUC: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    report = results['classification_report']
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision (Churn): {report['1']['precision']:.4f}")
    print(f"  Recall (Churn): {report['1']['recall']:.4f}")
    print(f"  F1-Score (Churn): {report['1']['f1-score']:.4f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    evaluate_model()