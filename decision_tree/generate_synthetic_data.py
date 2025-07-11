#!/usr/bin/env python3
"""Generate synthetic domain data for training the Dom2Vec classifier."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from sample_data
sys.path.append(str(Path(__file__).parent))

from sample_data import APPLICATION_DOMAINS
from synthetic_data.domain_synthesizer import DomainSynthesizer


def main():
    """Generate synthetic domain dataset."""
    print("=" * 60)
    print("Dom2Vec Synthetic Data Generator v1.1")
    print("=" * 60)
    print()
    
    # Configuration
    DOMAINS_PER_APP = 1000  # Target domains per application
    OUTPUT_DIR = Path("data")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize synthesizer
    print("Initializing domain synthesizer...")
    synthesizer = DomainSynthesizer(APPLICATION_DOMAINS, seed=42)
    
    print(f"\nAnalyzed {len(APPLICATION_DOMAINS)} applications:")
    for app, domains in APPLICATION_DOMAINS.items():
        print(f"  - {app}: {len(domains)} real domains")
    
    print(f"\nTarget: {DOMAINS_PER_APP} synthetic domains per application")
    print(f"Total target: {DOMAINS_PER_APP * len(APPLICATION_DOMAINS)} domains")
    print()
    
    # Generate synthetic data
    print("Generating synthetic domains...")
    print("-" * 40)
    synthetic_data = synthesizer.generate_synthetic_dataset(domains_per_app=DOMAINS_PER_APP)
    
    # Save synthetic data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    synthetic_path = OUTPUT_DIR / f"synthetic_domains_v1_{timestamp}.json"
    synthesizer.save_synthetic_data(synthetic_data, str(synthetic_path))
    
    # Create combined dataset (original + synthetic)
    print("\nCreating combined dataset...")
    combined_data = {}
    
    for app in APPLICATION_DOMAINS:
        # Always include all original domains
        combined_data[app] = APPLICATION_DOMAINS[app].copy()
        
        # Add synthetic domains
        if app in synthetic_data:
            combined_data[app].extend(synthetic_data[app])
            
        print(f"  {app}: {len(APPLICATION_DOMAINS[app])} real + {len(synthetic_data.get(app, []))} synthetic = {len(combined_data[app])} total")
    
    # Save combined dataset
    combined_path = OUTPUT_DIR / f"combined_dataset_v1_{timestamp}.json"
    with open(combined_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_domains': sum(len(domains) for domains in combined_data.values()),
                'real_domains': sum(len(domains) for domains in APPLICATION_DOMAINS.values()),
                'synthetic_domains': sum(len(domains) for domains in synthetic_data.values()),
                'applications': list(combined_data.keys()),
                'domains_per_app': {app: len(domains) for app, domains in combined_data.items()},
                'generation_timestamp': timestamp
            },
            'domains': combined_data
        }, f, indent=2)
    
    print(f"\nCombined dataset saved to {combined_path}")
    
    # Save just the original dataset for reference
    original_path = OUTPUT_DIR / "original_domains.json"
    with open(original_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_domains': sum(len(domains) for domains in APPLICATION_DOMAINS.values()),
                'applications': list(APPLICATION_DOMAINS.keys()),
                'domains_per_app': {app: len(domains) for app, domains in APPLICATION_DOMAINS.items()}
            },
            'domains': APPLICATION_DOMAINS
        }, f, indent=2)
    
    print(f"Original dataset saved to {original_path}")
    
    # Generate statistics report
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    total_synthetic = sum(len(domains) for domains in synthetic_data.values())
    total_combined = sum(len(domains) for domains in combined_data.values())
    
    print(f"\nSummary:")
    print(f"  - Original domains: {sum(len(domains) for domains in APPLICATION_DOMAINS.values())}")
    print(f"  - Synthetic domains: {total_synthetic}")
    print(f"  - Combined total: {total_combined}")
    print(f"  - Improvement factor: {total_combined / sum(len(domains) for domains in APPLICATION_DOMAINS.values()):.1f}x")
    
    # Show sample synthetic domains
    print("\nSample synthetic domains generated:")
    for app in list(synthetic_data.keys())[:3]:  # Show first 3 apps
        print(f"\n{app}:")
        for domain in synthetic_data[app][:5]:  # Show first 5 domains
            print(f"  - {domain}")
    
    print("\nNext steps:")
    print("1. Run 'python dom2vec_app_classifier.py --data data/combined_dataset_v1_*.json'")
    print("2. Compare performance with original dataset")
    print("3. Adjust generation parameters if needed")


if __name__ == "__main__":
    main()