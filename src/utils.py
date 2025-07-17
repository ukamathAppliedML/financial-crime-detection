"""
Financial Crime Detection Utility Functions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def load_fincrime_dataset(file_path: str) -> pd.DataFrame:
    """
    Load financial crime dataset from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load dataset from {file_path}: {str(e)}")


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the financial crime dataset structure and content
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    required_columns = ['conversation', 'label', 'category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation = {
        'valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'total_samples': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_labels': df['label'].unique().tolist() if 'label' in df.columns else [],
        'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
        'categories': df['category'].unique().tolist() if 'category' in df.columns else []
    }
    
    return validation


def prepare_data_splits(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                       random_state: int = 42) -> Tuple[List, List, List, List, List, List]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: Input DataFrame with 'conversation' and 'label' columns
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    texts = df['conversation'].tolist()
    labels = df['label'].tolist()
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: separate train and validation from remaining data
    val_ratio = val_size / (1 - test_size)  # Adjust validation ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive dataset analysis
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    # Basic statistics
    analysis = {
        'total_samples': len(df),
        'features': df.columns.tolist(),
        'missing_values': df.isnull().sum().sum()
    }
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        analysis['label_distribution'] = {
            'counts': label_counts.to_dict(),
            'percentages': (label_counts / len(df) * 100).to_dict()
        }
    
    if 'category' in df.columns:
        # Crime categories
        if 'label' in df.columns:
            suspicious_categories = df[df['label'] == 1]['category'].value_counts()
            normal_categories = df[df['label'] == 0]['category'].value_counts()
            
            analysis['suspicious_categories'] = suspicious_categories.to_dict()
            analysis['normal_categories'] = normal_categories.to_dict()
        else:
            analysis['all_categories'] = df['category'].value_counts().to_dict()
    
    if 'conversation' in df.columns:
        # Text statistics
        df['conversation_length'] = df['conversation'].str.len()
        df['word_count'] = df['conversation'].str.split().str.len()
        
        analysis['text_stats'] = {
            'avg_char_length': df['conversation_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'min_word_count': df['word_count'].min(),
            'max_word_count': df['word_count'].max(),
            'median_word_count': df['word_count'].median()
        }
    
    return analysis


def plot_dataset_overview(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create visualization plots for dataset overview
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Label distribution pie chart
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        axes[0,0].pie(label_counts.values, labels=['Normal', 'Suspicious'], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Label Distribution')
    
    # Word count distribution by label
    if 'conversation' in df.columns:
        df['word_count'] = df['conversation'].str.split().str.len()
        
        if 'label' in df.columns:
            df[df['label'] == 0]['word_count'].hist(alpha=0.7, bins=20, 
                                                   label='Normal', ax=axes[0,1])
            df[df['label'] == 1]['word_count'].hist(alpha=0.7, bins=20, 
                                                   label='Suspicious', ax=axes[0,1])
            axes[0,1].set_xlabel('Word Count')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Word Count Distribution by Label')
            axes[0,1].legend()
        else:
            df['word_count'].hist(bins=20, ax=axes[0,1])
            axes[0,1].set_xlabel('Word Count')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Word Count Distribution')
    
    # Category distributions
    if 'category' in df.columns and 'label' in df.columns:
        # Suspicious categories
        suspicious_categories = df[df['label'] == 1]['category'].value_counts()
        if len(suspicious_categories) > 0:
            suspicious_categories.plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Suspicious Activity Category Distribution')
            axes[1,0].set_xlabel('Category')
            axes[1,0].set_ylabel('Count')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Normal categories
        normal_categories = df[df['label'] == 0]['category'].value_counts()
        if len(normal_categories) > 0:
            normal_categories.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Normal Activity Category Distribution')
            axes[1,1].set_xlabel('Category')
            axes[1,1].set_ylabel('Count')
            axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive dataset summary
    
    Args:
        df: Input DataFrame
    """
    analysis = analyze_dataset(df)
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Features: {analysis['features']}")
    print(f"Missing values: {analysis['missing_values']}")
    
    if 'label_distribution' in analysis:
        print("\n=== LABEL DISTRIBUTION ===")
        for label, count in analysis['label_distribution']['counts'].items():
            percentage = analysis['label_distribution']['percentages'][label]
            label_name = "Normal" if label == 0 else "Suspicious"
            print(f"{label_name} ({label}): {count} ({percentage:.1f}%)")
    
    if 'suspicious_categories' in analysis:
        print("\n=== SUSPICIOUS ACTIVITY CATEGORIES ===")
        for category, count in analysis['suspicious_categories'].items():
            print(f"{category}: {count}")
    
    if 'normal_categories' in analysis:
        print("\n=== NORMAL ACTIVITY CATEGORIES ===")
        for category, count in analysis['normal_categories'].items():
            print(f"{category}: {count}")
    
    if 'text_stats' in analysis:
        print("\n=== TEXT LENGTH STATISTICS ===")
        stats = analysis['text_stats']
        print(f"Average conversation length: {stats['avg_char_length']:.1f} characters")
        print(f"Average word count: {stats['avg_word_count']:.1f} words")
        print(f"Min word count: {stats['min_word_count']}")
        print(f"Max word count: {stats['max_word_count']}")
        print(f"Median word count: {stats['median_word_count']:.1f}")


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality issues
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality report
    """
    quality_report = {
        'total_samples': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'empty_conversations': 0,
        'duplicate_conversations': 0,
        'label_issues': []
    }
    
    if 'conversation' in df.columns:
        # Check for empty conversations
        empty_conversations = (df['conversation'].str.strip() == '').sum()
        quality_report['empty_conversations'] = empty_conversations
        
        # Check for duplicate conversations
        duplicate_conversations = df['conversation'].duplicated().sum()
        quality_report['duplicate_conversations'] = duplicate_conversations
    
    if 'label' in df.columns:
        # Check label values
        unique_labels = df['label'].unique()
        expected_labels = [0, 1]
        unexpected_labels = [label for label in unique_labels if label not in expected_labels]
        if unexpected_labels:
            quality_report['label_issues'].append(f"Unexpected label values: {unexpected_labels}")
    
    return quality_report


# Example test cases for validation (AML examples)
TEST_EXAMPLES = [
    {
        "text": "Please execute the standard EUR/USD trade for our institutional client. All documentation is complete.",
        "expected": "Normal"
    },
    {
        "text": "Let's keep each transfer under $9,000 to avoid the paperwork. We can do multiple smaller ones.",
        "expected": "Suspicious"
    },
    {
        "text": "Route this through the Cayman entity first, then to Switzerland, then back here.",
        "expected": "Suspicious"
    },
    {
        "text": "Client needs emergency liquidity for planned acquisition closing tomorrow. Standard procedures apply.",
        "expected": "Normal"
    }
]