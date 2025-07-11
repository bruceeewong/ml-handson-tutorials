"""
Training Script for Dom2Vec Application Classifier
Provides easy interface to train and evaluate the model with different configurations.
"""

import argparse
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

from dom2vec_app_classifier import Dom2VecAppClassifier
from sample_data import get_all_training_data, print_data_summary
from domain_processor import DomainProcessor

def main():
    parser = argparse.ArgumentParser(description='Train Dom2Vec Application Classifier')
    
    # Model parameters
    parser.add_argument('--vector-size', type=int, default=100, 
                       help='Word2Vec embedding dimension (default: 100)')
    parser.add_argument('--window', type=int, default=5,
                       help='Word2Vec context window size (default: 5)')
    parser.add_argument('--min-count', type=int, default=1,
                       help='Minimum word frequency for Word2Vec (default: 1)')
    
    # Decision tree parameters
    parser.add_argument('--max-depth', type=int, default=15,
                       help='Maximum depth of decision tree (default: 15)')
    parser.add_argument('--min-samples-split', type=int, default=5,
                       help='Minimum samples required to split (default: 5)')
    parser.add_argument('--min-samples-leaf', type=int, default=2,
                       help='Minimum samples required at leaf (default: 2)')
    
    # Output options
    parser.add_argument('--save-model', type=str, default='dom2vec_model.pkl',
                       help='Path to save trained model (default: dom2vec_model.pkl)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for analysis')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Data options
    parser.add_argument('--test-domains', nargs='+', 
                       default=['netflix.com', 'api.spotify.com', 'teams.microsoft.com', 
                               'youtube.com', 'github.com'],
                       help='Domains to test after training')
    
    args = parser.parse_args()
    
    print("Dom2Vec Application Classifier Training")
    print("=" * 50)
    
    # Show data summary
    if args.verbose:
        print_data_summary()
        print()
    
    # Initialize classifier with specified parameters
    print(f"Initializing classifier with vector_size={args.vector_size}, window={args.window}")
    classifier = Dom2VecAppClassifier(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count
    )
    
    # Modify decision tree parameters
    def custom_train():
        """Custom training function with modified DT parameters."""
        training_data = get_all_training_data()
        print(f"Training on {len(training_data)} domain samples...")
        
        # Prepare sentences for Word2Vec
        sentences = classifier.prepare_training_sentences(training_data)
        
        # Train Word2Vec
        classifier.train_word2vec(sentences)
        
        # Extract features for decision tree
        print("Extracting features for decision tree training...")
        X = []
        y = []
        
        for domain, app_label in training_data:
            features = classifier.extract_all_features(domain)
            X.append(features)
            y.append(app_label)
        
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report, accuracy_score
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = classifier.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train decision tree with custom parameters
        print(f"Training decision tree with max_depth={args.max_depth}, "
              f"min_samples_split={args.min_samples_split}, "
              f"min_samples_leaf={args.min_samples_leaf}")
        
        classifier.decision_tree = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        )
        
        classifier.decision_tree.fit(X_train, y_train)
        classifier.prepare_feature_names()
        
        # Evaluate
        y_pred = classifier.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(classifier.decision_tree, X_train, y_train, cv=5)
        
        print(f"Training completed!")
        print(f"Test accuracy: {accuracy:.3f}")
        print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Detailed classification report
        y_test_labels = classifier.label_encoder.inverse_transform(y_test)
        y_pred_labels = classifier.label_encoder.inverse_transform(y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_labels': y_test_labels,
            'pred_labels': y_pred_labels
        }
    
    # Train the model
    start_time = time.time()
    print("\nStarting training...")
    
    try:
        metrics = custom_train()
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Test predictions on specified domains
        print("\n" + "="*60)
        print("Testing Predictions on Sample Domains:")
        print("="*60)
        
        for domain in args.test_domains:
            try:
                app, confidence = classifier.predict(domain)
                print(f"{domain:30} -> {app:18} (confidence: {confidence:.3f})")
            except Exception as e:
                print(f"{domain:30} -> Error: {e}")
        
        # Show feature importance
        print(f"\nTop 10 Most Important Features:")
        print("-" * 40)
        importance_df = classifier.get_feature_importance(10)
        for _, row in importance_df.iterrows():
            print(f"{row['feature']:25} {row['importance']:.4f}")
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating analysis plots...")
            
            try:
                # Feature importance plot
                classifier.plot_feature_importance(15)
                
                # Confusion matrix
                classifier.plot_confusion_matrix(metrics['test_labels'], metrics['pred_labels'])
                
                # Additional analysis plot - domain length distribution by app
                import pandas as pd
                import seaborn as sns
                
                # Analyze domain characteristics by application
                training_data = get_all_training_data()
                analysis_data = []
                
                for domain, app in training_data:
                    processor = DomainProcessor()
                    features = processor.get_domain_features(domain)
                    analysis_data.append({
                        'application': app,
                        'domain_length': features['domain_length'],
                        'subdomain_count': features['subdomain_count'],
                        'token_count': features['token_count']
                    })
                
                df = pd.DataFrame(analysis_data)
                
                # Plot domain length distribution
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=df, x='application', y='domain_length')
                plt.xticks(rotation=45)
                plt.title('Domain Length Distribution by Application')
                plt.tight_layout()
                plt.show()
                
                # Plot token count distribution
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=df, x='application', y='token_count')
                plt.xticks(rotation=45)
                plt.title('Token Count Distribution by Application')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
        
        # Save model
        if args.save_model:
            print(f"\nSaving model to {args.save_model}...")
            classifier.save_model(args.save_model)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Model accuracy: {metrics['accuracy']:.3f}")
        print(f"Cross-validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Total applications: {len(classifier.label_encoder.classes_)}")
        print(f"Word2Vec vocabulary size: {len(classifier.word2vec_model.wv.key_to_index)}")
        print(f"Decision tree depth: {classifier.decision_tree.get_depth()}")
        print(f"Model saved to: {args.save_model}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

def test_domain_processing():
    """Test domain processing functionality."""
    print("Testing Domain Processing...")
    print("-" * 30)
    
    processor = DomainProcessor()
    test_domains = [
        "netflix.com",
        "nflxvideo.net",
        "api.spotify.com", 
        "teams.microsoft.com",
        "fonts.googleapis.com",
        "assets-cdn.github.com"
    ]
    
    for domain in test_domains:
        tokens = processor.process_domain(domain)
        features = processor.get_domain_features(domain)
        print(f"\nDomain: {domain}")
        print(f"Tokens: {tokens}")
        print(f"Length: {features['domain_length']}, "
              f"Subdomains: {features['subdomain_count']}, ")

if __name__ == "__main__":
    # If no arguments provided, run a simple test
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running domain processing test...")
        test_domain_processing()
        print("\nTo train the model, run:")
        print("python train_model.py --plot --verbose")
    else:
        main() 