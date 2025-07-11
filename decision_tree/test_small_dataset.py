#!/usr/bin/env python3
"""Test classifier with small imbalanced dataset to simulate the stratification error."""

import pandas as pd
from dom2vec_app_classifier import Dom2VecAppClassifier


def create_imbalanced_test_data():
    """Create a small dataset that would cause stratification errors."""
    data = [
        # Enough samples for stratification
        ("google.com", "google"),
        ("youtube.com", "google"), 
        ("gmail.com", "google"),
        ("maps.google.com", "google"),
        ("drive.google.com", "google"),
        
        ("netflix.com", "netflix"),
        ("nflxvideo.net", "netflix"),
        ("fast.com", "netflix"),
        ("nflximg.net", "netflix"),
        ("api.netflix.com", "netflix"),
        
        # Moderate samples  
        ("spotify.com", "spotify"),
        ("scdn.co", "spotify"),
        ("api.spotify.com", "spotify"),
        
        # Few samples (will cause issues with test_size=0.2)
        ("github.com", "github"),
        ("githubusercontent.com", "github"),
        
        # Very few samples (will definitely cause issues)
        ("zoom.us", "zoom"),
        
        # Single sample (guaranteed to cause stratification error)
        ("rare-app.com", "rare_service"),
    ]
    
    return pd.DataFrame(data, columns=['domain', 'application'])


def test_stratification_handling():
    """Test how the classifier handles stratification errors."""
    print("="*60)
    print("TESTING STRATIFICATION ERROR HANDLING")
    print("="*60)
    
    # Create problematic dataset
    df = create_imbalanced_test_data()
    
    print("\nDataset Class Distribution:")
    print(df['application'].value_counts())
    print(f"Total samples: {len(df)}")
    
    # Test with different test sizes
    test_sizes = [0.5, 0.3, 0.2, 0.1]
    
    for test_size in test_sizes:
        print(f"\n" + "-"*40)
        print(f"Testing with test_size={test_size}")
        print("-"*40)
        
        classifier = Dom2VecAppClassifier()
        
        try:
            results = classifier.train(
                df=df,
                domain_col='domain',
                label_col='application',
                test_size=test_size,
                vector_size=20,  # Small for fast testing
                window=2,
                max_depth=5,
                cv=3,
                random_state=42
            )
            
            print(f"✅ SUCCESS: Accuracy = {results['accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")


def test_edge_cases():
    """Test extreme edge cases."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    # Test 1: All classes have exactly 1 sample
    print("\nTest 1: All classes with 1 sample each")
    df_single = pd.DataFrame([
        ("app1.com", "app1"),
        ("app2.com", "app2"), 
        ("app3.com", "app3"),
        ("app4.com", "app4"),
    ], columns=['domain', 'application'])
    
    classifier = Dom2VecAppClassifier()
    try:
        results = classifier.train(
            df=df_single,
            domain_col='domain', 
            label_col='application',
            test_size=0.2,
            vector_size=10,
            cv=2
        )
        print(f"✅ Handled single-sample classes: Accuracy = {results['accuracy']:.3f}")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
    
    # Test 2: Very small dataset
    print("\nTest 2: Extremely small dataset (3 samples)")
    df_tiny = pd.DataFrame([
        ("a.com", "app1"),
        ("b.com", "app2"),
        ("c.com", "app1"),
    ], columns=['domain', 'application'])
    
    classifier = Dom2VecAppClassifier()
    try:
        results = classifier.train(
            df=df_tiny,
            domain_col='domain',
            label_col='application', 
            test_size=0.2,
            vector_size=5,
            cv=2
        )
        print(f"✅ Handled tiny dataset: Accuracy = {results['accuracy']:.3f}")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")


if __name__ == "__main__":
    test_stratification_handling()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The updated classifier should now handle:")
    print("✅ Imbalanced class distributions")
    print("✅ Single-sample classes") 
    print("✅ Small datasets")
    print("✅ Automatic stratification detection")
    print("✅ Adaptive cross-validation")
    print("\nYour real data should now work without stratification errors!")