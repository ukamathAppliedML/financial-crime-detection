"""
Example Usage of Financial Crime Detection System

This script demonstrates how to use the financial crime detection package for:
1. Loading and analyzing data
2. Training a model
3. Making predictions
4. Evaluating performance

Currently demonstrated with AML (Anti-Money Laundering) use case.
"""

from src import (
    load_fincrime_dataset, 
    prepare_data_splits, 
    print_dataset_summary,
    FinCrimeModel,
    FinCrimePredictor, 
    FinCrimeEvaluator,
    TEST_EXAMPLES
)
import pandas as pd


def main():
    """Main example function"""
    
    # Example 1: Load and analyze dataset
    print("=== LOADING DATASET ===")
    # Replace with your actual dataset path
    dataset_path = "data/AntiMoneyLaundering-v4.csv"
    
    try:
        df = load_fincrime_dataset(dataset_path)
        print_dataset_summary(df)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using simulated AML data for demonstration...")
        # Create sample AML data for demo purposes
        df = pd.DataFrame({
            'conversation': [
                "Standard transaction processing for client account",
                "Keep transfers under 9k to avoid reporting requirements",
                "Route through offshore entities for privacy",
                "Regular business transaction with proper documentation",
                "Break this deposit into smaller amounts",
                "Client provided all necessary compliance documentation"
            ],
            'label': [0, 1, 1, 0, 1, 0],
            'category': [
                'Normal Business', 'Amount Threshold', 'Offshore Accounts', 
                'Normal Business', 'Structuring', 'Normal Business'
            ]
        })
        print("Created sample AML dataset for demonstration")
    
    # Example 2: Prepare data splits
    print("\n=== PREPARING DATA SPLITS ===")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(df)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Example 3: Train model (commented out as it requires GPU/significant time)
    print("\n=== MODEL TRAINING ===")
    print("Starting model training...")

    model = FinCrimeModel()
    training_results, test_dataset = model.train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir='./trained_model'
    )
    model.save_model('./trained_model')
    print("Training completed!")
    
    # Example 4: Load pre-trained model and make predictions
    print("\n=== MAKING PREDICTIONS ===")
    print("To use a trained model, you would do:")
    print("""
    predictor = FinCrimePredictor('path/to/trained/model')
    
    # Single prediction
    result = predictor.predict("Keep deposits under 10k to avoid reporting")
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.3f})")
    
    # Batch predictions
    texts = ["Normal transaction", "Suspicious activity"]
    results = predictor.predict_batch(texts)
    """)
    
    # Example 5: Test with sample examples
    print("\n=== TESTING WITH SAMPLE EXAMPLES ===")
    print("Here are AML test examples that would be used:")
    for i, example in enumerate(TEST_EXAMPLES, 1):
        print(f"{i}. Text: {example['text'][:60]}...")
        print(f"   Expected: {example['expected']}")
        print()
    
    # Example 6: Model evaluation
    print("\n=== MODEL EVALUATION ===")
    print("To evaluate a trained model:")
    print("""
    evaluator = FinCrimeEvaluator(predictor)
    evaluator.print_evaluation_summary(X_test, y_test)
    evaluator.plot_confusion_matrix(X_test, y_test)
    evaluator.plot_performance_metrics(X_test, y_test)
    
    # Find misclassified examples
    misclassified = evaluator.find_misclassified_examples(X_test, y_test)
    print(f"False Positives: {len(misclassified['false_positives'])}")
    print(f"False Negatives: {len(misclassified['false_negatives'])}")
    """)


def demo_prediction_simulation():
    """Simulate predictions for demonstration"""
    print("\n=== SIMULATED PREDICTION DEMO ===")
    
    for i, example in enumerate(TEST_EXAMPLES, 1):
        text = example['text']
        expected = example['expected']
        
        # Simulate prediction (in real usage, this would be from trained model)
        simulated_confidence = 0.85 if expected == "Suspicious" else 0.92
        
        print(f"Example {i}:")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Simulated Prediction: {expected} (Confidence: {simulated_confidence:.3f})")
        print("-" * 50)


def complete_workflow_example():
    """Example of complete training workflow"""
    print("\n=== COMPLETE WORKFLOW EXAMPLE ===")
    print("""
    # 1. Load and prepare AML data
    df = load_fincrime_dataset('data/AntiMoneyLaundering-v4.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(df)
    
    # 2. Initialize and train model
    model = FinCrimeModel(model_name="xlm-roberta-base")
    training_results, test_dataset = model.train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir='./fincrime_model'
    )
    
    # 3. Save the trained model
    model.save_model('./fincrime_model')
    
    # 4. Load model for prediction
    predictor = FinCrimePredictor('./fincrime_model')
    
    # 5. Make predictions on AML examples
    aml_examples = [
        "Keep each transfer under $9,000 to avoid paperwork",
        "Route this through the Cayman entity first",
        "Standard EUR/USD trade with proper documentation",
        "Break this large deposit into smaller amounts"
    ]
    
    for text in aml_examples:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.3f})")
    
    # 6. Evaluate model performance
    evaluator = FinCrimeEvaluator(predictor)
    metrics = evaluator.evaluate(X_test, y_test)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Suspicious Activity Detection F1: {metrics['f1_suspicious']:.4f}")
    
    # 7. Analyze results
    evaluator.print_evaluation_summary(X_test, y_test)
    """)


def dataset_format_example():
    """Show the expected dataset format"""
    print("\n=== DATASET FORMAT REQUIREMENTS ===")
    print("Your dataset should be a CSV with these columns:")
    print("""
    conversation,label,category
    "Standard transaction with proper documentation",0,"Normal Business"
    "Keep transfers under 10k to avoid reporting",1,"Amount Threshold"
    "Route through offshore accounts",1,"Offshore Accounts"
    "Regular client wire transfer",0,"Normal Business"
    """)
    
    print("Column descriptions:")
    print("- conversation: Text content of financial communications")
    print("- label: 0 (Normal/Legitimate) or 1 (Suspicious)")
    print("- category: Subcategory of the activity (e.g., 'Amount Threshold', 'Structuring')")


if __name__ == "__main__":
    main()
    demo_prediction_simulation()
    complete_workflow_example()
    dataset_format_example()
    
    print("\n=== SETUP INSTRUCTIONS ===")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Place your AML dataset in: data/AntiMoneyLaundering-v4.csv")
    print("3. Run training: python -c 'from src import FinCrimeModel; ...'")
    print("4. Use trained model: python -c 'from src import FinCrimePredictor; ...'")
    print("\nThe system is generic and can be adapted for other financial crime types")
    print("by using the same dataset format with different categories and content.")
    print("For detailed usage, see the README.md file.")