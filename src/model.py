"""
Financial Crime Model Training Module
"""
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


class FinCrimeDataset(Dataset):
    """Dataset class for financial crime text classification"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class FinCrimeModel:
    """Financial Crime Model Training and Management Class"""
    
    def __init__(self, model_name="xlm-roberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_model(self):
        """Initialize tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "Normal", 1: "Suspicious"},
            label2id={"Normal": 0, "Suspicious": 1}
        )
        print(f"Model: {self.model_name}")
        print(f"Model parameters: {self.model.num_parameters():,}")
        
    def tokenize_texts(self, texts):
        """Tokenize input texts"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def create_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create torch datasets from inputs"""
        train_encodings = self.tokenize_texts(X_train)
        val_encodings = self.tokenize_texts(X_val)
        test_encodings = self.tokenize_texts(X_test)
        
        train_dataset = FinCrimeDataset(train_encodings, y_train)
        val_dataset = FinCrimeDataset(val_encodings, y_val)
        test_dataset = FinCrimeDataset(test_encodings, y_test)
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def setup_training(self, train_dataset, val_dataset, output_dir='./results'):
        """Setup trainer with training arguments"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=6,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            seed=42,
            dataloader_drop_last=False,
            report_to=None,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, output_dir='./results'):
        """Complete training pipeline"""
        if self.model is None:
            self.initialize_model()
            
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Setup training
        self.setup_training(train_dataset, val_dataset, output_dir)
        
        # Train model
        print("Starting training...")
        training_results = self.trainer.train()
        
        print("Training completed!")
        print(f"Training time: {training_results.metrics['train_runtime']:.2f} seconds")
        print(f"Final training loss: {training_results.metrics['train_loss']:.4f}")
        
        return training_results, test_dataset
    
    def save_model(self, save_path):
        """Save trained model and tokenizer"""
        if self.trainer is not None:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model saved to: {save_path}")
        else:
            print("No trained model to save")
    
    def load_model(self, model_path):
        """Load pre-trained model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from: {model_path}")