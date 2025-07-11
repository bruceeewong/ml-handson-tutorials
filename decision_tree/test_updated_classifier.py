#!/usr/bin/env python3
"""Test the updated Dom2Vec classifier with DataFrame interface and synthetic data."""

import json
import pandas as pd
from pathlib import Path

from dom2vec_app_classifier import Dom2VecAppClassifier


def load_dataset_as_dataframe(data_path: str) -> pd.DataFrame:
    """Load dataset from JSON and convert to DataFrame."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract domains from the nested structure
    domains_dict = data.get('domains', data)
    
    # Convert to list of (domain, app) tuples
    rows = []
    for app_name, domain_list in domains_dict.items():
        for domain in domain_list:
            rows.append({'domain': domain, 'application': app_name})
    
    return pd.DataFrame(rows)


def test_with_original_data():
    """Test with original dataset."""
    print("="*60)
    print("TESTING WITH ORIGINAL DATASET")
    print("="*60)
    
    df = load_dataset_as_dataframe("data/original_domains.json")
    print(f"Loaded {len(df)} samples with {df['application'].nunique()} applications")
    
    classifier = Dom2VecAppClassifier()
    
    # Train with custom parameters
    results = classifier.train(
        df=df,
        domain_col='domain',
        label_col='application',
        vector_size=100,
        window=5,
        max_depth=15,
        test_size=0.2,
        random_state=42,
        cv=5
    )
    
    return results, classifier


def test_with_synthetic_data():
    """Test with combined synthetic dataset."""
    print("\n" + "="*60)
    print("TESTING WITH COMBINED DATASET (ORIGINAL + SYNTHETIC)")
    print("="*60)
    
    # Find the combined dataset file
    data_dir = Path("data")
    combined_files = list(data_dir.glob("combined_dataset_v1_*.json"))
    
    if not combined_files:
        print("No combined dataset found!")
        return None, None
    
    combined_path = combined_files[0]
    df = load_dataset_as_dataframe(str(combined_path))
    
    print(f"Loaded {len(df)} samples with {df['application'].nunique()} applications")
    print(f"Samples per application:")
    print(df['application'].value_counts().sort_index())
    
    classifier = Dom2VecAppClassifier()
    
    # Train with custom parameters optimized for larger dataset
    results = classifier.train(
        df=df,
        domain_col='domain',
        label_col='application',
        vector_size=100,  # Full embedding size
        window=5,
        max_depth=20,     # Deeper tree for more data
        min_samples_split=10,  # Higher threshold for splits
        min_samples_leaf=5,    # Higher threshold for leaves
        test_size=0.2,
        random_state=42,
        cv=5,
        workers=8  # More workers for faster training
    )
    
    return results, classifier


def test_hyperparameter_variations():
    """Test different hyperparameter combinations."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TESTING")
    print("="*60)
    
    # Load original dataset for quick testing
    df = load_dataset_as_dataframe("data/original_domains.json")
    
    # Define parameter combinations to test
    param_combinations = [
        {"vector_size": 50, "window": 3, "max_depth": 10, "name": "Light"},
        {"vector_size": 100, "window": 5, "max_depth": 15, "name": "Standard"},
        {"vector_size": 150, "window": 7, "max_depth": 20, "name": "Heavy"},
    ]
    
    results = {}
    
    for params in param_combinations:
        print(f"\nTesting {params['name']} configuration:")
        print(f"  vector_size={params['vector_size']}, window={params['window']}, max_depth={params['max_depth']}")
        
        classifier = Dom2VecAppClassifier()
        
        result = classifier.train(
            df=df,
            domain_col='domain',
            label_col='application',
            vector_size=params['vector_size'],
            window=params['window'],
            max_depth=params['max_depth'],
            test_size=0.2,
            random_state=42,
            cv=3  # Faster CV for testing
        )
        
        results[params['name']] = {
            'accuracy': result['accuracy'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'params': params
        }
        
        print(f"  Results: Accuracy={result['accuracy']:.3f}, CV={result['cv_mean']:.3f}±{result['cv_std']:.3f}")
    
    # Summary
    print(f"\n{'Configuration':<12} {'Accuracy':<10} {'CV Mean':<10} {'CV Std':<10}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<12} {result['accuracy']:<10.3f} {result['cv_mean']:<10.3f} {result['cv_std']:<10.3f}")
    
    return results


def main():
    """Run all tests."""
    print("Dom2Vec Classifier Testing Suite")
    print("="*60)
    
    # Test 1: Original dataset
    original_results, original_classifier = test_with_original_data()
    
    # Test 2: Synthetic dataset
    synthetic_results, synthetic_classifier = test_with_synthetic_data()
    
    # Test 3: Hyperparameter variations
    hyperparam_results = test_hyperparameter_variations()
    
    # Comparison summary
    if original_results and synthetic_results:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        
        print(f"Original Dataset:")
        print(f"  Accuracy: {original_results['accuracy']:.3f}")
        print(f"  CV: {original_results['cv_mean']:.3f} ± {original_results['cv_std']:.3f}")
        
        print(f"\nSynthetic Dataset:")
        print(f"  Accuracy: {synthetic_results['accuracy']:.3f}")
        print(f"  CV: {synthetic_results['cv_mean']:.3f} ± {synthetic_results['cv_std']:.3f}")
        
        improvement = synthetic_results['accuracy'] - original_results['accuracy']
        print(f"\nImprovement: {improvement:+.3f} ({improvement/original_results['accuracy']*100:+.1f}%)")
    
    # Test predictions on both models
    test_domains = [
        "netflix.com",
        "api.spotify.com",
        "teams.microsoft.com", 
        "youtube.com",
        "github.com",
        "cdn.amazonaws.com",
        "static.facebook.com"
    ]
    
    if synthetic_classifier:
        print(f"\nSample predictions from synthetic model:")
        for domain in test_domains:
            try:
                app, confidence = synthetic_classifier.predict(domain)
                print(f"  {domain:<25} -> {app:<15} ({confidence:.3f})")
            except Exception as e:
                print(f"  {domain:<25} -> Error: {e}")


if __name__ == "__main__":
    main()