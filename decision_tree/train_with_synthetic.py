#!/usr/bin/env python3
"""Train Dom2Vec classifier with synthetic dataset."""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dom2vec_app_classifier import Dom2VecAppClassifier


def load_dataset(data_path: str) -> dict:
    """Load dataset from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if 'domains' in data:
        return data['domains']
    else:
        return data


def prepare_training_data(domains_dict: dict):
    """Prepare training data in the format expected by the classifier."""
    training_data = []
    
    for app_name, domain_list in domains_dict.items():
        for domain in domain_list:
            training_data.append((domain, app_name))
    
    return training_data


def train_and_evaluate(data_path: str, vector_size: int = 100, verbose: bool = True):
    """Train model with dataset and return results."""
    
    print(f"Loading dataset from: {data_path}")
    domains_dict = load_dataset(data_path)
    
    # Print dataset statistics
    total_domains = sum(len(domains) for domains in domains_dict.values())
    print(f"\nDataset Statistics:")
    print(f"  Total domains: {total_domains}")
    print(f"  Applications: {len(domains_dict)}")
    for app, domains in domains_dict.items():
        print(f"    {app}: {len(domains)} domains")
    
    # Prepare training data in the format expected by classifier
    training_data = prepare_training_data(domains_dict)
    
    # Initialize classifier
    classifier = Dom2VecAppClassifier(vector_size=vector_size)
    
    print(f"\nTraining classifier with {vector_size}D embeddings...")
    print("-" * 50)
    
    # Train the model
    results = classifier.train(training_data)
    
    # Test predictions on sample domains
    test_domains = [
        "netflix.com",
        "api.spotify.com", 
        "teams.microsoft.com",
        "youtube.com",
        "github.com",
        "cdn.example.com"
    ]
    
    print("\n" + "="*60)
    print("Testing Predictions:")
    print("="*60)
    
    for domain in test_domains:
        try:
            app, confidence = classifier.predict(domain)
            print(f"{domain:25} -> {app:15} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"{domain:25} -> Error: {e}")
    
    # Show feature importance
    if verbose:
        print("\nTop 15 Most Important Features:")
        importance_df = classifier.get_feature_importance(15)
        print(importance_df.to_string(index=False))
    
    return results, classifier


def compare_datasets():
    """Compare original vs synthetic dataset performance."""
    print("="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    data_dir = Path("data")
    
    # Find the files
    original_path = data_dir / "original_domains.json"
    combined_files = list(data_dir.glob("combined_dataset_v1_*.json"))
    
    if not original_path.exists():
        print("Original dataset not found!")
        return
    
    if not combined_files:
        print("Combined dataset not found!")
        return
    
    combined_path = combined_files[0]  # Use the most recent
    
    results = {}
    
    # Train on original dataset
    print("\n1. TRAINING ON ORIGINAL DATASET")
    print("-" * 50)
    results['original'], _ = train_and_evaluate(str(original_path), verbose=False)
    
    # Train on combined dataset
    print("\n2. TRAINING ON COMBINED DATASET (ORIGINAL + SYNTHETIC)")
    print("-" * 50)
    results['combined'], _ = train_and_evaluate(str(combined_path), verbose=False)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nAccuracy Comparison:")
    print(f"  Original dataset:  {results['original']['accuracy']:.3f}")
    print(f"  Combined dataset:  {results['combined']['accuracy']:.3f}")
    print(f"  Improvement:       {results['combined']['accuracy'] - results['original']['accuracy']:+.3f}")
    
    print(f"\nCross-Validation Comparison:")
    print(f"  Original dataset:  {results['original']['cv_mean']:.3f} ± {results['original']['cv_std']:.3f}")
    print(f"  Combined dataset:  {results['combined']['cv_mean']:.3f} ± {results['combined']['cv_std']:.3f}")
    print(f"  Mean improvement:  {results['combined']['cv_mean'] - results['original']['cv_mean']:+.3f}")
    print(f"  Std improvement:   {results['original']['cv_std'] - results['combined']['cv_std']:+.3f} (lower is better)")
    
    # Calculate relative improvement
    rel_accuracy_improvement = (results['combined']['accuracy'] / results['original']['accuracy'] - 1) * 100
    rel_cv_improvement = (results['combined']['cv_mean'] / results['original']['cv_mean'] - 1) * 100
    
    print(f"\nRelative Improvements:")
    print(f"  Accuracy:     {rel_accuracy_improvement:+.1f}%")
    print(f"  CV Mean:      {rel_cv_improvement:+.1f}%")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train Dom2Vec classifier with synthetic data')
    parser.add_argument('--data', type=str, help='Path to dataset JSON file')
    parser.add_argument('--vector-size', type=int, default=100, help='Word2Vec vector size')
    parser.add_argument('--compare', action='store_true', help='Compare original vs combined datasets')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_datasets()
    elif args.data:
        train_and_evaluate(args.data, args.vector_size, args.verbose)
    else:
        print("Please specify --data <path> or --compare")
        print("Example: python train_with_synthetic.py --data data/combined_dataset_v1_*.json")
        print("Example: python train_with_synthetic.py --compare")


if __name__ == "__main__":
    main()