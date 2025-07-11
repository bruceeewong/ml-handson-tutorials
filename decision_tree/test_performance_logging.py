#!/usr/bin/env python3
"""Test the anonymized performance logging functionality."""

import pandas as pd
import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dom2vec_app_classifier import Dom2VecAppClassifier


def test_performance_logging():
    """Test performance logging with synthetic dataset."""
    print("="*60)
    print("TESTING ANONYMIZED PERFORMANCE LOGGING")
    print("="*60)
    
    # Load the combined dataset
    data_dir = Path("data")
    combined_files = list(data_dir.glob("combined_dataset_v1_*.json"))
    
    if not combined_files:
        print("No combined dataset found! Using sample data instead.")
        from sample_data import get_all_training_data
        training_data = get_all_training_data()
        df = pd.DataFrame(training_data, columns=['domain', 'application'])
    else:
        # Load the dataset
        with open(combined_files[0], 'r') as f:
            data = json.load(f)
        
        domains_dict = data.get('domains', data)
        rows = []
        for app_name, domain_list in domains_dict.items():
            for domain in domain_list[:100]:  # Limit to 100 per app for faster testing
                rows.append({'domain': domain, 'application': app_name})
        
        df = pd.DataFrame(rows)
    
    print(f"\nDataset loaded: {len(df)} samples, {df['application'].nunique()} applications")
    
    # Initialize classifier
    classifier = Dom2VecAppClassifier()
    
    # Train with various hyperparameter combinations
    param_combinations = [
        {
            "name": "Standard Configuration",
            "params": {
                "vector_size": 100,
                "window": 5,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "test_size": 0.2,
                "cv": 5
            }
        },
        {
            "name": "Light Configuration",
            "params": {
                "vector_size": 50,
                "window": 3,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "test_size": 0.3,
                "cv": 3
            }
        }
    ]
    
    for config in param_combinations:
        print(f"\n" + "-"*40)
        print(f"Training with: {config['name']}")
        print("-"*40)
        
        # Train the model
        results = classifier.train(
            df=df,
            domain_col='domain',
            label_col='application',
            random_state=42,
            **config['params']
        )
        
        # Get the performance log
        performance_log = results['performance_log']
        
        # Save the log
        log_file = classifier.save_performance_log(performance_log)
        
        # Display formatted summary
        summary = classifier.format_performance_summary(performance_log)
        print("\n" + summary)
        
        # Also save the summary to a text file
        summary_file = log_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_file}")
        
        # Show how to share just the essential info
        print("\n" + "="*60)
        print("SHAREABLE PERFORMANCE METRICS:")
        print("="*60)
        print(f"Configuration: {config['name']}")
        print(f"Test Accuracy: {performance_log['overall_performance']['test_accuracy']:.3f}")
        print(f"CV Mean ± Std: {performance_log['overall_performance']['cv_mean_accuracy']:.3f} ± "
              f"{performance_log['overall_performance']['cv_std_accuracy']:.3f}")
        print(f"Macro F1 Score: {performance_log['overall_performance']['macro_f1']:.3f}")
        print(f"Dataset Size: {performance_log['dataset_info']['total_samples']} samples")
        print(f"Valid Classes: {performance_log['dataset_info']['total_classes']}")
        print(f"Filtered Classes: {performance_log['dataset_info']['filtered_classes']}")
        
        # Show anonymized confusion matrix snippet
        print("\nTop 3 Classes by F1 Score (Anonymized):")
        for metrics in performance_log['per_class_performance'][:3]:
            print(f"  {metrics['class']}: F1={metrics['f1_score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")


def demonstrate_log_structure():
    """Show the structure of the performance log."""
    print("\n" + "="*60)
    print("PERFORMANCE LOG STRUCTURE")
    print("="*60)
    
    example_structure = {
        "run_id": "unique_id",
        "timestamp": "ISO format timestamp",
        "model_configuration": {
            "model_type": "Dom2Vec + DecisionTree",
            "word2vec_params": {"vector_size": 100, "window": 5},
            "decision_tree_params": {"max_depth": 15},
            "training_params": {"test_size": 0.2, "cv_folds": 5}
        },
        "dataset_info": {
            "total_samples": "number",
            "total_classes": "number",
            "filtered_classes": "number",
            "class_distribution": {"app_000": "count", "app_001": "count"},
            "vocabulary_size": "number"
        },
        "overall_performance": {
            "test_accuracy": "float",
            "cv_mean_accuracy": "float",
            "cv_std_accuracy": "float",
            "macro_f1": "float",
            "weighted_f1": "float"
        },
        "per_class_performance": [
            {"class": "app_000", "precision": "float", "recall": "float", "f1_score": "float", "support": "int"}
        ],
        "confusion_matrix": {
            "labels": ["app_000", "app_001"],
            "matrix": [["true_positives", "false_positives"], ["false_negatives", "true_negatives"]]
        },
        "notes": {
            "anonymized": True,
            "class_mapping_hash": "hash for verification",
            "filtered_class_count": "number",
            "filtering_reason": "explanation"
        }
    }
    
    print(json.dumps(example_structure, indent=2))
    
    print("\n" + "="*60)
    print("KEY FEATURES:")
    print("="*60)
    print("✅ All application names are anonymized (app_000, app_001, etc.)")
    print("✅ Complete hyperparameter tracking for reproducibility")
    print("✅ Detailed per-class metrics for analysis")
    print("✅ Dataset statistics without revealing sensitive information")
    print("✅ Unique run ID and timestamp for tracking experiments")
    print("✅ Hash of class mapping for consistency verification")


if __name__ == "__main__":
    test_performance_logging()
    demonstrate_log_structure()
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. After training, access the log: results['performance_log']")
    print("2. Save to file: classifier.save_performance_log(log)")
    print("3. Format for sharing: classifier.format_performance_summary(log)")
    print("4. Share the anonymized JSON or text summary")
    print("\nThe logs contain no sensitive application names!")