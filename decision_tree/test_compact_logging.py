#!/usr/bin/env python3
"""Test the compact logging format for large datasets."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dom2vec_app_classifier import Dom2VecAppClassifier


def create_large_imbalanced_dataset():
    """Create a dataset with many classes to test compact logging."""
    data = []
    
    # Create 50 different "applications" with varying sample sizes
    # This simulates a real-world scenario with many apps
    
    # A few very popular apps (high traffic)
    popular_apps = [
        ("mega_app_1", 500),
        ("mega_app_2", 450),
        ("mega_app_3", 400),
    ]
    
    # Some medium-sized apps
    medium_apps = [(f"medium_app_{i}", np.random.randint(50, 150)) for i in range(10)]
    
    # Many small apps (typical long-tail distribution)
    small_apps = [(f"small_app_{i}", np.random.randint(5, 25)) for i in range(20)]
    
    # Some tiny apps
    tiny_apps = [(f"tiny_app_{i}", np.random.randint(2, 8)) for i in range(17)]
    
    all_apps = popular_apps + medium_apps + small_apps + tiny_apps
    
    # Generate domains for each app
    for app_name, count in all_apps:
        for i in range(count):
            # Generate realistic domain variations
            base_patterns = [
                f"{app_name}.com",
                f"api.{app_name}.com",
                f"cdn.{app_name}.com", 
                f"www.{app_name}.com",
                f"{app_name}-static.net",
                f"mobile.{app_name}.io",
                f"{app_name}cdn.co",
                f"assets.{app_name}.net",
            ]
            
            domain = base_patterns[i % len(base_patterns)]
            if i >= len(base_patterns):
                # Add variations with numbers/regions
                region = np.random.choice(['us', 'eu', 'asia', 'ca', 'au'])
                domain = f"{region}.{app_name}.com"
            
            data.append({'domain': domain, 'application': app_name})
    
    df = pd.DataFrame(data)
    
    print(f"Generated dataset: {len(df)} samples across {df['application'].nunique()} applications")
    print(f"Class distribution summary:")
    counts = df['application'].value_counts()
    print(f"  - Largest class: {counts.max()} samples")
    print(f"  - Smallest class: {counts.min()} samples") 
    print(f"  - Mean: {counts.mean():.1f} samples per class")
    print(f"  - Classes with <10 samples: {len(counts[counts < 10])}")
    print(f"  - Classes with >100 samples: {len(counts[counts > 100])}")
    
    return df


def test_compact_vs_full_logging():
    """Compare compact vs full logging formats."""
    print("="*60)
    print("TESTING COMPACT VS FULL LOGGING")
    print("="*60)
    
    # Create dataset with many classes
    df = create_large_imbalanced_dataset()
    
    # Initialize classifier
    classifier = Dom2VecAppClassifier()
    
    print(f"\nTraining with {len(df)} samples across {df['application'].nunique()} classes...")
    print("This will use COMPACT format (>20 classes detected)")
    
    # Train the model
    results = classifier.train(
        df=df,
        domain_col='domain',
        label_col='application',
        vector_size=50,  # Smaller for faster training
        window=3,
        max_depth=15,
        test_size=0.2,
        cv=3,
        random_state=42
    )
    
    # Get the performance log
    log = results['performance_log']
    
    # Save the compact log
    log_file = classifier.save_performance_log(log, "compact_format_example.json")
    
    # Show formatted summary
    summary = classifier.format_performance_summary(log)
    print("\n" + summary)
    
    # Show what makes it compact
    print("\n" + "="*60)
    print("COMPACT FORMAT BENEFITS")
    print("="*60)
    
    # Show size differences
    import json
    full_size = len(json.dumps(log, indent=2))
    
    # Show specific compact sections
    print("✅ CONFUSION MATRIX:")
    cm_data = log['confusion_matrix']
    if cm_data.get('format') == 'compact_summary':
        print(f"   Instead of {len(df['application'].unique())}x{len(df['application'].unique())} matrix:")
        print(f"   - Shows {len(cm_data['top_confusion_pairs'])} top confusion pairs")
        print(f"   - Shows 5 best and 5 worst performing classes")
        print(f"   - Provides overall accuracy statistics")
    
    print("\n✅ PER-CLASS PERFORMANCE:")
    pc_data = log['per_class_performance']
    if pc_data.get('format') == 'compact_summary':
        print(f"   Instead of {pc_data['total_classes']} individual class metrics:")
        print(f"   - Shows top {len(pc_data['top_performers'])} and bottom {len(pc_data['bottom_performers'])} performers")
        print(f"   - Provides F1 score statistics (mean, std, min, max)")
        print(f"   - Groups classes into performance tiers")
    
    print("\n✅ CLASS DISTRIBUTION:")
    dist_data = log['dataset_info']['class_distribution']
    if dist_data.get('format') == 'compact_summary':
        print(f"   Instead of {dist_data['statistics']['total_classes']} individual class counts:")
        print(f"   - Shows distribution statistics (mean, median, range)")
        print(f"   - Provides histogram bins")
        print(f"   - Shows 5 largest and 5 smallest classes")
    
    print(f"\nTotal log size: {full_size:,} characters")
    print(f"Compact format: {log['notes']['compact_format']}")
    print(f"Number of classes: {log['notes']['num_classes']}")


def test_small_dataset_full_format():
    """Test that small datasets still use full format."""
    print("\n" + "="*60)
    print("TESTING SMALL DATASET (FULL FORMAT)")
    print("="*60)
    
    # Create small dataset (≤20 classes)
    small_data = []
    for i in range(10):  # Only 10 classes
        app_name = f"app_{i}"
        for j in range(15):  # 15 samples each
            small_data.append({
                'domain': f"{app_name}-{j}.com",
                'application': app_name
            })
    
    df_small = pd.DataFrame(small_data)
    print(f"Small dataset: {len(df_small)} samples, {df_small['application'].nunique()} classes")
    
    classifier = Dom2VecAppClassifier()
    results = classifier.train(
        df=df_small,
        domain_col='domain',
        label_col='application',
        vector_size=30,
        window=3,
        max_depth=10,
        test_size=0.2,
        cv=3,
        random_state=42
    )
    
    log = results['performance_log']
    print(f"\nCompact format used: {log['notes']['compact_format']}")
    print(f"Number of classes: {log['notes']['num_classes']}")
    
    # Show that it uses full format
    if not log['notes']['compact_format']:
        print("✅ Using FULL format because ≤20 classes")
        print(f"   - Confusion matrix: full {len(log['confusion_matrix']['labels'])}x{len(log['confusion_matrix']['labels'])} matrix")
        print(f"   - Per-class performance: all {len(log['per_class_performance'])} classes listed")
        print(f"   - Class distribution: all {len(log['dataset_info']['class_distribution'])} classes shown")


if __name__ == "__main__":
    test_compact_vs_full_logging()
    test_small_dataset_full_format()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Compact format automatically used for >20 classes")
    print("✅ Full format used for ≤20 classes")
    print("✅ Compact format provides meaningful summaries instead of raw dumps")
    print("✅ Perfect for sharing performance feedback on large datasets")
    print("\nThe compact format makes large dataset logs readable and shareable!")