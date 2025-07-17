"""Visualization utilities for Financial Crime detection model."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import confusion_matrix


class ModelVisualizer:
    """Visualization class for Financial Crime detection model."""
    
    def __init__(self, figsize: tuple = (15, 12)):
        """Initialize visualizer."""
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_data_overview(self, data_stats: Dict[str, Any]) -> None:
        """Plot data overview including distributions and statistics."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Label distribution pie chart
        label_counts = data_stats['label_distribution']
        axes[0, 0].pie(
            label_counts.values(),
            labels=['Normal', 'Financial Crime'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightblue', 'lightcoral']
        )
        axes[0, 0].set_title('Label Distribution')
        
        # Category distribution for Financial Crime
        fincrime_categories = data_stats['fincrime_categories']
        if fincrime_categories:
            top_fincrime = dict(list(fincrime_categories.items())[:10])
            axes[0, 1].bar(range(len(top_fincrime)), list(top_fincrime.values()))
            axes[0, 1].set_title('Top 10 Financial Crime Categories')
            axes[0, 1].set_xlabel('Category')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(len(top_fincrime)))
            axes[0, 1].set_xticklabels(list(top_fincrime.keys()), rotation=45, ha='right')
        
        # Category distribution for Normal
        normal_categories = data_stats['normal_categories']
        if normal_categories:
            top_normal = dict(list(normal_categories.items())[:10])
            axes[1, 0].bar(range(len(top_normal)), list(top_normal.values()))
            axes[1, 0].set_title('Top 10 Normal Categories')
            axes[1, 0].set_xlabel('Category')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(top_normal)))
            axes[1, 0].set_xticklabels(list(top_normal.keys()), rotation=45, ha='right')
        
        # Text length statistics
        axes[1, 1].text(0.1, 0.8, f"Total Samples: {data_stats['total_samples']}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Avg Conversation Length: {data_stats['avg_conversation_length']:.1f} chars", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Avg Word Count: {data_stats['avg_word_count']:.1f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"Min Word Count: {data_stats['min_word_count']}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"Max Word Count: {data_stats['max_word_count']}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Text Statistics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_word_count_distribution(self, df: pd.DataFrame) -> None:
        """Plot word count distribution by label."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate word counts if not already present
        if 'word_count' not in df.columns:
            df['word_count'] = df['conversation'].str.split().str.len()
        
        # Plot histograms
        df[df['label'] == 0]['word_count'].hist(
            alpha=0.7, bins=20, label='Normal', ax=ax, color='lightblue'
        )
        df[df['label'] == 1]['word_count'].hist(
            alpha=0.7, bins=20, label='Financial Crime', ax=ax, color='lightcoral'
        )
        
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Word Count Distribution by Label')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Plot performance metrics as bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Performance metrics bar chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('fincrime_precision', metrics.get('precision', 0)),
            metrics.get('fincrime_recall', metrics.get('recall', 0)),
            metrics.get('fincrime_f1', metrics.get('f1', 0))
        ]
        
        bars = axes[0, 0].bar(metric_names, metric_values, 
                             color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrix heatmap
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Financial Crime'],
                       yticklabels=['Normal', 'Financial Crime'], ax=axes[0, 1])
            axes[0, 1].set_title('Confusion Matrix')
            axes[0, 1].set_ylabel('True Label')
            axes[0, 1].set_xlabel('Predicted Label')
        
        # Detailed metrics text
        if 'true_negatives' in metrics:
            axes[1, 0].text(0.1, 0.8, f"True Negatives: {metrics['true_negatives']}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.7, f"False Positives: {metrics['false_positives']}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"False Negatives: {metrics['false_negatives']}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.5, f"True Positives: {metrics['true_positives']}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.3, f"Specificity: {metrics.get('specificity', 0):.4f}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.2, f"Sensitivity: {metrics.get('sensitivity', 0):.4f}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Detailed Metrics')
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].axis('off')
        
        # Prediction distribution comparison
        if 'true_negatives' in metrics:
            true_counts = [metrics['true_negatives'] + metrics['false_negatives'],
                          metrics['true_positives'] + metrics['false_positives']]
            pred_counts = [metrics['true_negatives'] + metrics['false_positives'],
                          metrics['true_positives'] + metrics['false_negatives']]
            
            x = ['Normal', 'Financial Crime']
            width = 0.35
            x_pos = np.arange(len(x))
            
            axes[1, 1].bar(x_pos - width/2, true_counts, width, label='True', alpha=0.8)
            axes[1, 1].bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
            axes[1, 1].set_title('True vs Predicted Distribution')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(x)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, trainer_state) -> None:
        """Plot training history if available."""
        if not hasattr(trainer_state, 'log_history'):
            print("Training history not available.")
            return
        
        train_losses = [log['train_loss'] for log in trainer_state.log_history if 'train_loss' in log]
        eval_losses = [log['eval_loss'] for log in trainer_state.log_history if 'eval_loss' in log]
        
        if not train_losses or not eval_losses:
            print("No training/evaluation losses found in history.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps_train = list(range(len(train_losses)))
        steps_eval = list(range(len(eval_losses)))
        
        ax.plot(steps_train, train_losses, label='Training Loss', marker='o')
        ax.plot(steps_eval, eval_losses, label='Validation Loss', marker='s')
        ax.set_title('Training History')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_all_plots(self, data_stats: Dict[str, Any], metrics: Dict[str, float], 
                      df: pd.DataFrame, save_dir: str = './plots') -> None:
        """Save all plots to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Data overview
        plt.figure(figsize=self.figsize)
        self.plot_data_overview(data_stats)
        plt.savefig(f"{save_dir}/data_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Word count distribution
        plt.figure(figsize=(10, 6))
        self.plot_word_count_distribution(df)
        plt.savefig(f"{save_dir}/word_count_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics
        plt.figure(figsize=self.figsize)
        self.plot_performance_metrics(metrics)
        plt.savefig(f"{save_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All plots saved to {save_dir}")