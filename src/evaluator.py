"""
Financial Crime Model Evaluation Module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from typing import List, Dict, Tuple, Optional
from .predictor import FinCrimePredictor


class FinCrimeEvaluator:
    """Evaluation class for financial crime models"""
    
    def __init__(self, predictor: FinCrimePredictor):
        """
        Initialize evaluator with a predictor
        
        Args:
            predictor: Trained FinCrimePredictor instance
        """
        self.predictor = predictor
    
    def evaluate(self, texts: List[str], true_labels: List[int]) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            texts: List of test texts
            true_labels: List of true labels (0 for Normal, 1 for Suspicious)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions
        predictions = []
        confidences = []
        
        for text in texts:
            result = self.predictor.predict(text)
            predictions.append(result['class_id'])
            confidences.append(result['confidence'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_suspicious = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_suspicious = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_suspicious = 2 * (precision_suspicious * recall_suspicious) / (precision_suspicious + recall_suspicious) if (precision_suspicious + recall_suspicious) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'precision_suspicious': precision_suspicious,
            'recall_suspicious': recall_suspicious,
            'f1_suspicious': f1_suspicious,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def generate_classification_report(self, texts: List[str], true_labels: List[int]) -> str:
        """
        Generate detailed classification report
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
            
        Returns:
            Classification report as string
        """
        predictions = [self.predictor.predict(text)['class_id'] for text in texts]
        return classification_report(true_labels, predictions, target_names=['Normal', 'Suspicious'])
    
    def find_misclassified_examples(self, texts: List[str], true_labels: List[int]) -> Dict:
        """
        Find and analyze misclassified examples
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
            
        Returns:
            Dictionary with false positives and false negatives
        """
        results = []
        for text, true_label in zip(texts, true_labels):
            prediction = self.predictor.predict(text)
            results.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': prediction['class_id'],
                'confidence': prediction['confidence'],
                'correct': true_label == prediction['class_id']
            })
        
        results_df = pd.DataFrame(results)
        
        # False Positives (Normal classified as Suspicious)
        false_positives = results_df[
            (results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)
        ]
        
        # False Negatives (Suspicious classified as Normal)
        false_negatives = results_df[
            (results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)
        ]
        
        return {
            'false_positives': false_positives.to_dict('records'),
            'false_negatives': false_negatives.to_dict('records'),
            'all_results': results_df.to_dict('records')
        }
    
    def plot_confusion_matrix(self, texts: List[str], true_labels: List[int], 
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
            figsize: Figure size tuple
        """
        predictions = [self.predictor.predict(text)['class_id'] for text in texts]
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Suspicious'],
                   yticklabels=['Normal', 'Suspicious'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_performance_metrics(self, texts: List[str], true_labels: List[int],
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot performance metrics bar chart
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
            figsize: Figure size tuple
        """
        metrics = self.evaluate(texts, true_labels)
        
        metric_names = ['Accuracy', 'Precision (Suspicious)', 'Recall (Suspicious)', 'F1-Score (Suspicious)']
        metric_values = [
            metrics['accuracy'],
            metrics['precision_suspicious'],
            metrics['recall_suspicious'],
            metrics['f1_suspicious']
        ]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(metric_names, metric_values, 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        plt.title('Financial Crime Detection Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def print_evaluation_summary(self, texts: List[str], true_labels: List[int]) -> None:
        """
        Print comprehensive evaluation summary
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
        """
        metrics = self.evaluate(texts, true_labels)
        misclassified = self.find_misclassified_examples(texts, true_labels)
        
        print("=== FINANCIAL CRIME MODEL EVALUATION SUMMARY ===")
        print(f"Total test samples: {len(texts)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Suspicious Activity Detection Precision: {metrics['precision_suspicious']:.4f}")
        print(f"Suspicious Activity Detection Recall: {metrics['recall_suspicious']:.4f}")
        print(f"Suspicious Activity Detection F1-Score: {metrics['f1_suspicious']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Positives: {metrics['true_positives']}")
        
        print(f"\nMisclassification Analysis:")
        print(f"False Positives (Normal → Suspicious): {len(misclassified['false_positives'])}")
        print(f"False Negatives (Suspicious → Normal): {len(misclassified['false_negatives'])}")
        
        # Show sample misclassifications
        if misclassified['false_positives']:
            print("\nSample False Positives:")
            for i, fp in enumerate(misclassified['false_positives'][:3]):
                print(f"  {i+1}. {fp['text'][:100]}...")
        
        if misclassified['false_negatives']:
            print("\nSample False Negatives:")
            for i, fn in enumerate(misclassified['false_negatives'][:3]):
                print(f"  {i+1}. {fn['text'][:100]}...")


def evaluate_model(model_path: str, test_texts: List[str], test_labels: List[int]) -> Dict:
    """
    Convenience function to evaluate a model
    
    Args:
        model_path: Path to saved model
        test_texts: List of test texts
        test_labels: List of test labels
        
    Returns:
        Evaluation metrics dictionary
    """
    predictor = FinCrimePredictor(model_path)
    evaluator = FinCrimeEvaluator(predictor)
    return evaluator.evaluate(test_texts, test_labels)