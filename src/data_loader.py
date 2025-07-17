"""Data loading and preprocessing utilities for Financial Crime detection model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Dict, Any


class FinCrimeDataset(Dataset):
    """Custom dataset class for Financial Crime data."""
    
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        """Initialize dataset with encodings and labels."""
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)


class DataLoader:
    """Data loader class for Financial Crime dataset."""
    
    def __init__(self, data_path: str, tokenizer, config):
        """Initialize data loader."""
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully from {self.data_path}")
            print(f"Dataset shape: {self.df.shape}")
            return self.df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def validate_data(self) -> None:
        """Validate dataset structure and content."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Check required columns
        required_columns = ['conversation', 'label', 'category']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"Warning: Found null values:\n{null_counts}")
        
        # Check label distribution
        label_counts = self.df['label'].value_counts()
        print(f"Label distribution: {label_counts.to_dict()}")
        
        # Validate labels are binary (0 and 1)
        unique_labels = self.df['label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Text statistics
        self.df['conversation_length'] = self.df['conversation'].str.len()
        self.df['word_count'] = self.df['conversation'].str.split().str.len()
        
        # Label statistics
        label_counts = self.df['label'].value_counts()
        
        # Category statistics
        fincrime_categories = self.df[self.df['label'] == 1]['category'].value_counts()
        normal_categories = self.df[self.df['label'] == 0]['category'].value_counts()
        
        stats = {
            'total_samples': len(self.df),
            'columns': self.df.columns.tolist(),
            'label_distribution': label_counts.to_dict(),
            'fincrime_categories': fincrime_categories.to_dict(),
            'normal_categories': normal_categories.to_dict(),
            'avg_conversation_length': self.df['conversation_length'].mean(),
            'avg_word_count': self.df['word_count'].mean(),
            'min_word_count': self.df['word_count'].min(),
            'max_word_count': self.df['word_count'].max(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return stats
    
    def print_data_overview(self) -> None:
        """Print comprehensive data overview."""
        stats = self.get_data_statistics()
        
        print("=== DATASET OVERVIEW ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Features: {stats['columns']}")
        print(f"Missing values: {sum(stats['missing_values'].values())}")
        
        print("\n=== LABEL DISTRIBUTION ===")
        for label, count in stats['label_distribution'].items():
            percentage = count / stats['total_samples'] * 100
            label_name = "Normal" if label == 0 else "AML"
            print(f"{label_name} ({label}): {count} ({percentage:.1f}%)")
        
        print("\n=== TEXT LENGTH STATISTICS ===")
        print(f"Average conversation length: {stats['avg_conversation_length']:.1f} characters")
        print(f"Average word count: {stats['avg_word_count']:.1f} words")
        print(f"Min word count: {stats['min_word_count']}")
        print(f"Max word count: {stats['max_word_count']}")
        
        print("\n=== AML CATEGORY DISTRIBUTION ===")
        for category, count in list(stats['aml_categories'].items())[:10]:  # Top 10
            print(f"{category}: {count}")
        
        print("\n=== NORMAL CATEGORY DISTRIBUTION ===")
        for category, count in list(stats['normal_categories'].items())[:10]:  # Top 10
            print(f"{category}: {count}")
    
    def split_data(self) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """Split data into train, validation, and test sets."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        texts = self.df['conversation'].tolist()
        labels = self.df['label'].tolist()
        
        # First split: 80% train, 20% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=labels
        )
        
        # Second split: Split the 20% into 10% validation and 10% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
        
        print(f"\nTrain label distribution: {np.bincount(y_train)}")
        print(f"Validation label distribution: {np.bincount(y_val)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize list of texts."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
    
    def create_datasets(self) -> Tuple[FinCrimeDataset, FinCrimeDataset, FinCrimeDataset]:
        """Create train, validation, and test datasets."""
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        
        # Tokenize texts
        print("Tokenizing texts...")
        train_encodings = self.tokenize_texts(X_train)
        val_encodings = self.tokenize_texts(X_val)
        test_encodings = self.tokenize_texts(X_test)
        
        print("Tokenization completed!")
        print(f"Train encodings shape: {train_encodings['input_ids'].shape}")
        print(f"Validation encodings shape: {val_encodings['input_ids'].shape}")
        print(f"Test encodings shape: {test_encodings['input_ids'].shape}")
        
        # Create datasets
        train_dataset = FinCrimeDataset(train_encodings, y_train)
        val_dataset = FinCrimeDataset(val_encodings, y_val)
        test_dataset = FinCrimeDataset(test_encodings, y_test)
        
        return train_dataset, val_dataset, test_dataset