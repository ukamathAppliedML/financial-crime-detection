"""
Financial Crime Prediction Module
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union


class FinCrimePredictor:
    """Financial Crime Text Classifier for predicting suspicious communications"""
    
    def __init__(self, model_path: str):
        """
        Initialize the financial crime predictor with a trained model
        
        Args:
            model_path: Path to the saved model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Financial crime model loaded successfully from {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict if a single text is normal or crime-related
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction label and confidence score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        label = "Suspicious" if predicted_class == 1 else "Normal"
        
        return {
            "label": label,
            "confidence": confidence,
            "class_id": predicted_class
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict for a batch of texts
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_with_probabilities(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict with detailed probability scores for both classes
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction, confidence, and probability breakdown
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Get probabilities for both classes
            prob_normal = predictions[0][0].item()
            prob_suspicious = predictions[0][1].item()
        
        label = "Suspicious" if predicted_class == 1 else "Normal"
        
        return {
            "label": label,
            "confidence": confidence,
            "class_id": predicted_class,
            "probabilities": {
                "Normal": prob_normal,
                "Suspicious": prob_suspicious
            }
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_path": self.model_path,
            "model_type": self.model.config.model_type,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.tokenizer.model_max_length,
            "device": str(self.device)
        }


def load_predictor(model_path: str) -> FinCrimePredictor:
    """
    Convenience function to load a financial crime predictor
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Initialized FinCrimePredictor instance
    """
    return FinCrimePredictor(model_path)