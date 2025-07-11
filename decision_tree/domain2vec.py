import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Domain2Vec:
    """
    Simplified Domain2Vec implementation for converting domain names to vectors
    that can be used with decision trees to capture domain similarity.
    """
    
    def __init__(self, vector_dim=50, use_pca=True):
        self.vector_dim = vector_dim
        self.use_pca = use_pca
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=1000,
            lowercase=True
        )
        self.pca = PCA(n_components=vector_dim) if use_pca else None
        self.scaler = StandardScaler()
        self.domain_vectors = {}
        self.fitted = False
    
    def _extract_domain_features(self, domains):
        """Extract various features from domain names"""
        features = []
        
        for domain in domains:
            # Parse domain
            parsed = urlparse(f"http://{domain}")
            domain_name = parsed.netloc or domain
            
            # Basic features
            feature_dict = {
                'length': len(domain_name),
                'num_dots': domain_name.count('.'),
                'num_hyphens': domain_name.count('-'),
                'num_digits': sum(c.isdigit() for c in domain_name),
                'num_vowels': sum(c.lower() in 'aeiou' for c in domain_name),
                'num_consonants': sum(c.isalpha() and c.lower() not in 'aeiou' for c in domain_name),
            }
            
            # TLD features
            parts = domain_name.split('.')
            if len(parts) > 1:
                tld = parts[-1].lower()
                feature_dict.update({
                    'is_com': 1 if tld == 'com' else 0,
                    'is_org': 1 if tld == 'org' else 0,
                    'is_net': 1 if tld == 'net' else 0,
                    'is_edu': 1 if tld == 'edu' else 0,
                    'is_gov': 1 if tld == 'gov' else 0,
                    'tld_length': len(tld)
                })
            else:
                feature_dict.update({
                    'is_com': 0, 'is_org': 0, 'is_net': 0,
                    'is_edu': 0, 'is_gov': 0, 'tld_length': 0
                })
            
            # Subdomain features
            feature_dict['num_subdomains'] = max(0, len(parts) - 2)
            
            # Character distribution features
            char_counts = Counter(domain_name.lower())
            most_common = char_counts.most_common(3)
            for i, (char, count) in enumerate(most_common):
                feature_dict[f'top_char_{i+1}'] = ord(char) if char.isalnum() else 0
                feature_dict[f'top_char_{i+1}_count'] = count
            
            # Pad if less than 3 most common characters
            for i in range(len(most_common), 3):
                feature_dict[f'top_char_{i+1}'] = 0
                feature_dict[f'top_char_{i+1}_count'] = 0
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def fit(self, domains, domain_labels=None):
        """
        Fit the Domain2Vec model on a list of domains
        
        Args:
            domains: List of domain names
            domain_labels: Optional labels for domains (for supervised learning)
        """
        print(f"Fitting Domain2Vec on {len(domains)} domains...")
        
        # Extract traditional features
        traditional_features = self._extract_domain_features(domains)
        
        # Extract TF-IDF features from domain strings
        tfidf_features = self.tfidf_vectorizer.fit_transform(domains)
        tfidf_dense = tfidf_features.toarray()
        
        # Combine features
        combined_features = np.hstack([
            traditional_features.values,
            tfidf_dense
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Apply PCA if requested
        if self.use_pca and scaled_features.shape[1] > self.vector_dim:
            final_features = self.pca.fit_transform(scaled_features)
        else:
            final_features = scaled_features
        
        # Store domain vectors
        for i, domain in enumerate(domains):
            self.domain_vectors[domain] = final_features[i]
        
        self.fitted = True
        print(f"Domain2Vec fitted. Vector dimension: {final_features.shape[1]}")
        
        return final_features
    
    def transform(self, domains):
        """Transform new domains to vectors"""
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
        
        # Extract features for new domains
        traditional_features = self._extract_domain_features(domains)
        tfidf_features = self.tfidf_vectorizer.transform(domains)
        tfidf_dense = tfidf_features.toarray()
        
        # Combine features
        combined_features = np.hstack([
            traditional_features.values,
            tfidf_dense
        ])
        
        # Scale features
        scaled_features = self.scaler.transform(combined_features)
        
        # Apply PCA if used during fitting
        if self.use_pca:
            final_features = self.pca.transform(scaled_features)
        else:
            final_features = scaled_features
        
        return final_features
    
    def get_domain_similarity(self, domain1, domain2):
        """Calculate similarity between two domains"""
        if domain1 not in self.domain_vectors or domain2 not in self.domain_vectors:
            # Transform domains if not in stored vectors
            domains_to_transform = []
            if domain1 not in self.domain_vectors:
                domains_to_transform.append(domain1)
            if domain2 not in self.domain_vectors:
                domains_to_transform.append(domain2)
            
            if domains_to_transform:
                new_vectors = self.transform(domains_to_transform)
                for i, domain in enumerate(domains_to_transform):
                    self.domain_vectors[domain] = new_vectors[i]
        
        vec1 = self.domain_vectors[domain1].reshape(1, -1)
        vec2 = self.domain_vectors[domain2].reshape(1, -1)
        
        return cosine_similarity(vec1, vec2)[0][0]
    
    def find_similar_domains(self, target_domain, n_similar=5):
        """Find most similar domains to target domain"""
        if target_domain not in self.domain_vectors:
            # Transform target domain
            target_vector = self.transform([target_domain])[0]
            self.domain_vectors[target_domain] = target_vector
        
        target_vec = self.domain_vectors[target_domain].reshape(1, -1)
        similarities = []
        
        for domain, vector in self.domain_vectors.items():
            if domain != target_domain:
                sim = cosine_similarity(target_vec, vector.reshape(1, -1))[0][0]
                similarities.append((domain, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]


class DomainDecisionTree:
    """
    Decision Tree classifier that uses Domain2Vec embeddings
    """
    
    def __init__(self, domain2vec_params=None, tree_params=None):
        self.domain2vec = Domain2Vec(**(domain2vec_params or {}))
        self.tree = DecisionTreeClassifier(**(tree_params or {'max_depth': 10, 'random_state': 42}))
        self.fitted = False
    
    def fit(self, domains, labels):
        """
        Fit the model on domains and their labels
        
        Args:
            domains: List of domain names
            labels: List of labels (e.g., 'malicious', 'benign', 'phishing', etc.)
        """
        print("Training Domain Decision Tree...")
        
        # Get domain vectors
        domain_vectors = self.domain2vec.fit(domains)
        
        # Train decision tree
        self.tree.fit(domain_vectors, labels)
        self.fitted = True
        
        print("Training completed!")
    
    def predict(self, domains):
        """Predict labels for new domains"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        domain_vectors = self.domain2vec.transform(domains)
        return self.tree.predict(domain_vectors)
    
    def predict_proba(self, domains):
        """Predict probabilities for new domains"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        domain_vectors = self.domain2vec.transform(domains)
        return self.tree.predict_proba(domain_vectors)
    
    def get_feature_importance(self):
        """Get feature importance from the decision tree"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        return self.tree.feature_importances_
    
    def evaluate_model(self, test_domains, test_labels, detailed=True):
        """
        Comprehensive evaluation of the decision tree model
        
        Args:
            test_domains: List of test domain names
            test_labels: List of true labels for test domains
            detailed: Whether to print detailed evaluation metrics
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        predictions = self.predict(test_domains)
        probabilities = self.predict_proba(test_domains)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Get per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )
        
        # Create confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'class_names': self.tree.classes_
        }
        
        if detailed:
            self._print_evaluation_report(results, test_domains, test_labels)
        
        return results
    
    def _print_evaluation_report(self, results, test_domains, test_labels):
        """Print detailed evaluation report"""
        print("\n" + "="*50)
        print("DECISION TREE MODEL EVALUATION REPORT")
        print("="*50)
        
        print(f"\nOverall Performance:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Weighted Precision: {results['precision_weighted']:.3f}")
        print(f"Weighted Recall: {results['recall_weighted']:.3f}")
        print(f"Weighted F1-Score: {results['f1_weighted']:.3f}")
        
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(results['class_names']):
            print(f"{class_name}:")
            print(f"  Precision: {results['precision_per_class'][i]:.3f}")
            print(f"  Recall: {results['recall_per_class'][i]:.3f}")
            print(f"  F1-Score: {results['f1_per_class'][i]:.3f}")
            print(f"  Support: {results['support_per_class'][i]}")
        
        print(f"\nClassification Report:")
        print(classification_report(test_labels, results['predictions'], 
                                  target_names=results['class_names']))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Show some prediction examples
        print(f"\nSample Predictions:")
        for i in range(min(10, len(test_domains))):
            confidence = np.max(results['probabilities'][i])
            print(f"Domain: {test_domains[i]}")
            print(f"  True: {test_labels[i]} | Predicted: {results['predictions'][i]} | Confidence: {confidence:.3f}")
    
    def cross_validate(self, domains, labels, cv_folds=5):
        """
        Perform cross-validation on the dataset
        
        Args:
            domains: List of domain names
            labels: List of labels
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary containing cross-validation results
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        # Get domain vectors
        domain_vectors = self.domain2vec.fit(domains)
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.tree, domain_vectors, labels, cv=cv, scoring='accuracy')
        
        # Calculate additional metrics for each fold
        precision_scores = cross_val_score(self.tree, domain_vectors, labels, cv=cv, scoring='precision_weighted')
        recall_scores = cross_val_score(self.tree, domain_vectors, labels, cv=cv, scoring='recall_weighted')
        f1_scores = cross_val_score(self.tree, domain_vectors, labels, cv=cv, scoring='f1_weighted')
        
        results = {
            'accuracy_scores': cv_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }
        
        print(f"Cross-Validation Results:")
        print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
        print(f"Precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")
        print(f"Recall: {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")
        print(f"F1-Score: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
        
        return results


def demo_domain2vec():
    """Comprehensive demonstration and evaluation of Domain2Vec functionality"""
    
    # Expanded sample domains for different categories
    sample_domains = {
        'legitimate': [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'netflix.com', 'twitter.com', 'linkedin.com',
            'github.com', 'stackoverflow.com', 'youtube.com', 'wikipedia.org',
            'reddit.com', 'cnn.com', 'bbc.com', 'nytimes.com',
            'spotify.com', 'instagram.com', 'dropbox.com', 'salesforce.com'
        ],
        'suspicious': [
            'g00gle.com', 'faceb00k.com', 'amaz0n.com', 'micr0soft.com',
            'app1e.com', 'netf1ix.com', 'tw1tter.com', 'l1nkedin.com',
            'g1thub.com', 'stack0verflow.com', 'y0utube.com', 'w1kipedia.org',
            'redd1t.com', 'cnn-news.com', 'bbc-live.com', 'ny-times.com',
            'sp0tify.com', '1nstagram.com', 'dr0pbox.com', 'sales-force.com'
        ],
        'malicious': [
            'secure-bank-update.com', 'paypal-verification.net', 'amazon-security.org',
            'facebook-login.info', 'google-account.biz', 'microsoft-update.co',
            'apple-support.tk', 'netflix-billing.ml', 'twitter-security.ga',
            'linkedin-verify.cf', 'github-security.tk', 'youtube-premium.ml',
            'instagram-verify.ga', 'dropbox-storage.tk', 'reddit-gold.cf',
            'cnn-breaking.info', 'bbc-urgent.org', 'spotify-free.biz',
            'stackoverflow-jobs.tk', 'wikipedia-donate.ml'
        ],
        'random': [
            'xyzabc123.com', 'random-site.org', 'test123.net', 'example-domain.com',
            'sample-website.org', 'demo-site.net', 'placeholder.com', 'temp-domain.org',
            'foobar-example.com', 'dummy-site.info', 'test-website.biz', 'sample123.co',
            'random-string.net', 'placeholder123.org', 'example-test.com', 'demo123.info',
            'temp-site.net', 'filler-domain.org', 'dummy123.com', 'test-example.biz'
        ]
    }
    
    # Flatten domains and create labels
    all_domains = []
    all_labels = []
    
    for category, domains in sample_domains.items():
        all_domains.extend(domains)
        all_labels.extend([category] * len(domains))
    
    print("=== Enhanced Domain2Vec Demonstration & Evaluation ===\n")
    print(f"Total domains: {len(all_domains)}")
    print(f"Categories: {list(sample_domains.keys())}")
    print(f"Domains per category: {[len(domains) for domains in sample_domains.values()]}")
    
    # Split data into train and test sets
    train_domains, test_domains, train_labels, test_labels = train_test_split(
        all_domains, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    print(f"\nTrain set: {len(train_domains)} domains")
    print(f"Test set: {len(test_domains)} domains")
    
    # Initialize and fit Domain2Vec
    print("\n=== Domain2Vec Training ===")
    d2v = Domain2Vec(vector_dim=25)
    vectors = d2v.fit(train_domains)
    
    print(f"Generated vectors shape: {vectors.shape}")
    print(f"Sample vector for 'google.com': {d2v.domain_vectors['google.com'][:5]}...\n")
    
    # Test similarity analysis
    print("=== Domain Similarity Analysis ===")
    test_pairs = [
        ('google.com', 'g00gle.com'),
        ('facebook.com', 'faceb00k.com'),
        ('amazon.com', 'secure-bank-update.com'),
        ('google.com', 'xyzabc123.com'),
        ('paypal-verification.net', 'amazon-security.org')
    ]
    
    for domain1, domain2 in test_pairs:
        if domain1 in d2v.domain_vectors and domain2 in d2v.domain_vectors:
            similarity = d2v.get_domain_similarity(domain1, domain2)
            print(f"Similarity between '{domain1}' and '{domain2}': {similarity:.3f}")
    
    print("\n=== Finding Similar Domains ===")
    if 'google.com' in d2v.domain_vectors:
        similar_domains = d2v.find_similar_domains('google.com', n_similar=5)
        print(f"Domains most similar to 'google.com':")
        for domain, sim in similar_domains:
            print(f"  {domain}: {sim:.3f}")
    
    # Train Decision Tree with comprehensive evaluation
    print("\n=== Decision Tree Training & Evaluation ===")
    dt_model = DomainDecisionTree(
        domain2vec_params={'vector_dim': 20},
        tree_params={'max_depth': 12, 'min_samples_split': 5, 'random_state': 42}
    )
    
    # Fit the model
    dt_model.fit(train_domains, train_labels)
    
    # Comprehensive evaluation on test set
    evaluation_results = dt_model.evaluate_model(test_domains, test_labels, detailed=True)
    
    # Feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    feature_importance = dt_model.get_feature_importance()
    
    # Plot feature importance (top 15 features)
    top_indices = np.argsort(feature_importance)[-15:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_indices)), feature_importance[top_indices])
    plt.ylabel('Feature Index')
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    print(f"Top 10 most important features (indices): {np.argsort(feature_importance)[-10:]}")
    print(f"Feature importance range: [{np.min(feature_importance):.4f}, {np.max(feature_importance):.4f}]")
    
    # Cross-validation
    cv_results = dt_model.cross_validate(all_domains, all_labels, cv_folds=5)
    
    # Additional test predictions on new domains
    print("\n=== Additional Test Predictions ===")
    additional_test_domains = [
        'g0ogle.com',           # typosquatting
        'paypal-urgent.com',    # phishing-like
        'legitimate-site.com',  # generic legitimate
        'microsoft-365.org',    # suspicious
        'random-test123.net'    # random
    ]
    
    additional_predictions = dt_model.predict(additional_test_domains)
    additional_probabilities = dt_model.predict_proba(additional_test_domains)
    
    for i, domain in enumerate(additional_test_domains):
        pred = additional_predictions[i]
        confidence = np.max(additional_probabilities[i])
        print(f"{domain} -> {pred} (confidence: {confidence:.3f})")
    
    # Visualize domain embeddings
    print("\n=== Visualizing Domain Embeddings ===")
    
    # Use all domains for visualization
    all_vectors = d2v.fit(all_domains)
    
    # Use t-SNE for 2D visualization
    if all_vectors.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_domains)-1))
        vectors_2d = tsne.fit_transform(all_vectors)
    else:
        vectors_2d = all_vectors
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    colors = {
        'legitimate': 'blue', 
        'suspicious': 'orange', 
        'malicious': 'red', 
        'random': 'green'
    }
    
    for i, (domain, label) in enumerate(zip(all_domains, all_labels)):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], 
                   c=colors[label], alpha=0.7, s=60)
        
        # Only annotate a subset to avoid clutter
        if i % 4 == 0:  # Show every 4th domain name
            plt.annotate(domain, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(3, 3), textcoords='offset points', 
                        fontsize=7, alpha=0.8)
    
    plt.title('Domain2Vec Embeddings Visualization (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Create legend
    for label, color in colors.items():
        plt.scatter([], [], c=color, alpha=0.7, s=60, label=f'{label} ({sum(1 for l in all_labels if l == label)})')
    plt.legend(title='Domain Categories')
    
    plt.tight_layout()
    plt.show()
    
    # Summary of evaluation
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset Size: {len(all_domains)} domains across {len(sample_domains)} categories")
    print(f"Train/Test Split: {len(train_domains)}/{len(test_domains)}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.3f}")
    print(f"Cross-Validation Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"Weighted F1-Score: {evaluation_results['f1_weighted']:.3f}")
    print(f"Cross-Validation F1-Score: {cv_results['f1_mean']:.3f} ± {cv_results['f1_std']:.3f}")
    
    return d2v, dt_model, evaluation_results, cv_results

if __name__ == "__main__":
    # Run comprehensive demonstration and evaluation
    domain2vec_model, decision_tree_model, eval_results, cv_results = demo_domain2vec()
    
    print("\n=== Advanced Usage Examples ===")
    print("# Initialize Domain2Vec with custom parameters")
    print("d2v = Domain2Vec(vector_dim=30, use_pca=True)")
    print("vectors = d2v.fit(your_domains)")
    print("")
    print("# Get domain similarity")
    print("similarity = d2v.get_domain_similarity('domain1.com', 'domain2.com')")
    print("similar_domains = d2v.find_similar_domains('target.com', n_similar=5)")
    print("")
    print("# Train decision tree for domain classification")
    print("dt = DomainDecisionTree(")
    print("    domain2vec_params={'vector_dim': 20},")
    print("    tree_params={'max_depth': 10, 'min_samples_split': 5}")
    print(")")
    print("dt.fit(train_domains, train_labels)")
    print("")
    print("# Comprehensive evaluation")
    print("eval_results = dt.evaluate_model(test_domains, test_labels)")
    print("cv_results = dt.cross_validate(all_domains, all_labels)")
    print("")
    print("# Get predictions with confidence")
    print("predictions = dt.predict(new_domains)")
    print("probabilities = dt.predict_proba(new_domains)")
    print("feature_importance = dt.get_feature_importance()")
    
    print(f"\n=== Final Model Performance Summary ===")
    print(f"Model achieved {eval_results['accuracy']:.1%} accuracy on test set")
    print(f"Cross-validation shows {cv_results['accuracy_mean']:.1%} ± {cv_results['accuracy_std']:.1%} accuracy")
    print(f"This indicates {'good' if eval_results['accuracy'] > 0.8 else 'moderate' if eval_results['accuracy'] > 0.6 else 'poor'} performance for domain classification")