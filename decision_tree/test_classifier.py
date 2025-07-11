"""
Test Script for Dom2Vec Application Classifier
Provides easy interface to test trained models and analyze predictions.
"""

import argparse
import sys
from pathlib import Path

from dom2vec_app_classifier import Dom2VecAppClassifier
from domain_processor import DomainProcessor

def load_and_test_model(model_path: str, test_domains: list):
    """Load model and test on provided domains."""
    try:
        # Load the trained model
        print(f"Loading model from {model_path}...")
        classifier = Dom2VecAppClassifier()
        classifier.load_model(model_path)
        
        print(f"Model loaded successfully!")
        print(f"Available applications: {list(classifier.label_encoder.classes_)}")
        print(f"Word2Vec vocabulary size: {len(classifier.word2vec_model.wv.key_to_index)}")
        print()
        
        # Test predictions
        print("Testing Predictions:")
        print("=" * 60)
        
        results = []
        for domain in test_domains:
            try:
                app, confidence = classifier.predict(domain)
                results.append((domain, app, confidence))
                print(f"{domain:35} -> {app:18} (confidence: {confidence:.3f})")
            except Exception as e:
                results.append((domain, "ERROR", 0.0))
                print(f"{domain:35} -> ERROR: {e}")
        
        return classifier, results
        
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found!")
        print("Please train a model first using: python train_model.py")
        return None, []
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, []

def interactive_mode(classifier):
    """Interactive mode for testing domains."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter domain names to classify (type 'quit' to exit)")
    print("Examples: netflix.com, api.spotify.com, teams.microsoft.com")
    print()
    
    while True:
        try:
            domain = input("Domain: ").strip()
            
            if domain.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not domain:
                continue
            
            # Get prediction
            app, confidence = classifier.predict(domain)
            
            # Get domain analysis
            processor = DomainProcessor()
            tokens = processor.process_domain(domain)
            features = processor.get_domain_features(domain)
            
            print(f"\nResults for: {domain}")
            print(f"  Predicted App: {app}")
            print(f"  Confidence:    {confidence:.3f}")
            print(f"  Tokens:        {tokens}")
            print(f"  Domain Length: {features['domain_length']}")
            print(f"  Subdomains:    {features['subdomain_count']}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def analyze_domain_tokens(domain: str):
    """Analyze how a domain gets tokenized."""
    processor = DomainProcessor()
    
    print(f"\nDomain Analysis for: {domain}")
    print("-" * 40)
    
    # Clean domain
    clean_domain = processor.clean_domain(domain)
    print(f"Cleaned:     {clean_domain}")
    
    # Split by separators
    separators = processor.split_by_separators(clean_domain)
    print(f"Separators:  {separators}")
    
    # Final tokens
    tokens = processor.process_domain(domain)
    print(f"Final tokens: {tokens}")
    
    # Features
    features = processor.get_domain_features(domain)
    print(f"\nKey Features:")
    for key, value in features.items():
        if isinstance(value, bool):
            if value:  # Only show True values for clarity
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

def batch_test_from_file(classifier, file_path: str):
    """Test domains from a file (one domain per line)."""
    try:
        with open(file_path, 'r') as f:
            domains = [line.strip() for line in f if line.strip()]
        
        print(f"Testing {len(domains)} domains from {file_path}")
        print("=" * 60)
        
        results = {}
        for domain in domains:
            try:
                app, confidence = classifier.predict(domain)
                if app not in results:
                    results[app] = []
                results[app].append((domain, confidence))
            except Exception as e:
                print(f"Error processing {domain}: {e}")
        
        # Print results grouped by application
        for app, domain_list in results.items():
            print(f"\n{app.upper()}:")
            for domain, conf in sorted(domain_list, key=lambda x: x[1], reverse=True):
                print(f"  {domain:35} (confidence: {conf:.3f})")
                
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")

def main():
    parser = argparse.ArgumentParser(description='Test Dom2Vec Application Classifier')
    
    parser.add_argument('--model', type=str, default='dom2vec_model.pkl',
                       help='Path to trained model file (default: dom2vec_model.pkl)')
    parser.add_argument('--domains', nargs='+',
                       help='Domains to test')
    parser.add_argument('--file', type=str,
                       help='File containing domains to test (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--analyze', type=str,
                       help='Analyze token extraction for a specific domain')
    
    args = parser.parse_args()
    
    # Default test domains if none provided
    if not args.domains and not args.file and not args.analyze:
        args.domains = [
            'netflix.com',
            'nflxvideo.net',
            'api.spotify.com',
            'scdn.co',
            'teams.microsoft.com',
            'youtube.com',
            'googlevideo.com',
            'github.com',
            'amazonaws.com',
            'apple.com',
            'facebook.com',
            'instagram.com',
            'unknown-service.example.com'
        ]
    
    print("Dom2Vec Application Classifier - Test Script")
    print("=" * 50)
    
    # Handle domain analysis mode
    if args.analyze:
        analyze_domain_tokens(args.analyze)
        return
    
    # Load and test model
    classifier, results = load_and_test_model(args.model, args.domains or [])
    
    if classifier is None:
        return
    
    # Handle file testing
    if args.file:
        batch_test_from_file(classifier, args.file)
    
    # Start interactive mode if requested
    if args.interactive:
        interactive_mode(classifier)
    
    # Print some statistics if we tested domains
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        # Count predictions by application
        app_counts = {}
        total_confidence = 0
        
        for domain, app, conf in results:
            if app != "ERROR":
                app_counts[app] = app_counts.get(app, 0) + 1
                total_confidence += conf
        
        print(f"Total domains tested: {len(results)}")
        print(f"Average confidence: {total_confidence/len([r for r in results if r[1] != 'ERROR']):.3f}")
        print(f"\nPredictions by application:")
        
        for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {app:20}: {count:3} domains")

def quick_test():
    """Quick test function for development."""
    print("Quick Test Mode - Testing Domain Processing")
    print("-" * 40)
    
    test_domains = [
        "netflix.com",
        "nflxvideo.net", 
        "api.spotify.com",
        "teams.microsoft.com",
        "fonts.googleapis.com",
        "assets-cdn.github.com",
        "audio-ak-spotify-com.akamaized.net",
        "mortiscontrastatim.com",  # Dictionary-based example
        "cvyh1po636avyrsxebwbkn7.ddns.net"  # Random example
    ]
    
    processor = DomainProcessor()
    
    for domain in test_domains:
        print(f"\nDomain: {domain}")
        tokens = processor.process_domain(domain)
        features = processor.get_domain_features(domain)
        
        print(f"  Tokens: {tokens}")
        print(f"  Length: {features['domain_length']}, "
              f"Subdomains: {features['subdomain_count']}, "
              f"Token count: {features['token_count']}")
        
        # Show key patterns detected
        patterns = []
        if features['has_api_pattern']: patterns.append('API')
        if features['has_cdn_pattern']: patterns.append('CDN')
        if features['has_netflix_pattern']: patterns.append('Netflix')
        if features['has_spotify_pattern']: patterns.append('Spotify')
        if features['has_google_pattern']: patterns.append('Google')
        if features['has_microsoft_pattern']: patterns.append('Microsoft')
        
        if patterns:
            print(f"  Patterns: {', '.join(patterns)}")

if __name__ == "__main__":
    # If no arguments provided, run quick test
    if len(sys.argv) == 1:
        quick_test()
        print("\nTo test with a trained model:")
        print("python test_classifier.py --model dom2vec_model.pkl --interactive")
    else:
        main() 