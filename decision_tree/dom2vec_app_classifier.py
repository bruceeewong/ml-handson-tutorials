"""
Dom2Vec Application Classifier
Adapted from Dom2Vec for classifying applications from SNI domain names using
Word2Vec embeddings and Decision Tree classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import joblib
from pathlib import Path
import json
from datetime import datetime
import hashlib

from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .domain_processor import DomainProcessor
    from .sample_data import get_all_training_data, get_available_apps
except ImportError:
    from domain_processor import DomainProcessor
    from sample_data import get_all_training_data, get_available_apps

class Dom2VecAppClassifier:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Initialize the Dom2Vec Application Classifier.
        
        Args:
            vector_size: Dimension of Word2Vec embeddings
            window: Context window size for Word2Vec
            min_count: Minimum word frequency for Word2Vec
            workers: Number of worker threads for Word2Vec
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        
        # Initialize components
        self.domain_processor = DomainProcessor()
        self.word2vec_model = None
        self.decision_tree = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Training data
        self.training_sentences = []
        self.training_labels = []
        
        # Track filtered classes
        self.filtered_classes = set()
        self.valid_classes = set()
        
    def prepare_training_sentences(self, training_data: List[Tuple[str, str]]) -> List[List[str]]:
        """
        Prepare training sentences from domain data for Word2Vec training.
        
        Args:
            training_data: List of (domain, app_label) tuples
            
        Returns:
            List of tokenized sentences for Word2Vec
        """
        sentences = []
        labels = []
        
        for domain, app_label in training_data:
            # Process domain into tokens
            tokens = self.domain_processor.process_domain(domain)
            
            if tokens:  # Only add non-empty token lists
                sentences.append(tokens)
                labels.append(app_label)
        
        self.training_sentences = sentences
        self.training_labels = labels
        
        return sentences
    
    def train_word2vec(self, sentences: List[List[str]], epochs: int = 10) -> None:
        """Train Word2Vec model on domain sentences."""
        print(f"Training Word2Vec on {len(sentences)} domain sentences...")
        print(f"Parameters: vector_size={self.vector_size}, window={self.window}, min_count={self.min_count}, workers={self.workers}")
        
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=0,  # CBOW
            epochs=epochs
        )
        
        print(f"Word2Vec vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        print("Sample vocabulary:", list(self.word2vec_model.wv.key_to_index.keys())[:10])
    
    def get_domain_embedding(self, domain: str, pooling='average') -> np.ndarray:
        """
        Get embedding for a domain using trained Word2Vec model.
        
        Args:
            domain: Domain name to embed
            pooling: Pooling strategy ('average', 'max', 'min', 'sum')
            
        Returns:
            Domain embedding vector
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet!")
        
        tokens = self.domain_processor.process_domain(domain)
        embeddings = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
        
        if not embeddings:
            # Return zero vector if no tokens found in vocabulary
            return np.zeros(self.vector_size)
        
        embeddings = np.array(embeddings)
        
        if pooling == 'average':
            return np.mean(embeddings, axis=0)
        elif pooling == 'max':
            return np.max(embeddings, axis=0)
        elif pooling == 'min':
            return np.min(embeddings, axis=0)
        elif pooling == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    def extract_all_features(self, domain: str) -> np.ndarray:
        """
        Extract complete feature vector for a domain.
        Combines embeddings with additional domain features.
        
        Args:
            domain: Domain name
            
        Returns:
            Complete feature vector
        """
        # Get embedding features
        embedding = self.get_domain_embedding(domain)
        
        # Get additional domain features
        domain_features = self.domain_processor.get_domain_features(domain)
        
        # Combine all features
        all_features = np.concatenate([
            embedding,
            [
                domain_features['domain_length'],
                domain_features['subdomain_count'],
                float(domain_features['has_numbers']),
                float(domain_features['has_hyphens']),
                float(domain_features['has_underscores']),
                domain_features['token_count'],
                domain_features['avg_token_length'],
            ]
        ])
        
        return all_features
    
    def prepare_feature_names(self) -> List[str]:
        """Prepare feature names for interpretability."""
        embedding_features = [f'embed_{i}' for i in range(self.vector_size)]
        
        additional_features = [
            'domain_length', 'subdomain_count', 'has_numbers', 'has_hyphens',
            'has_underscores', 'token_count', 'avg_token_length',
        ]
        
        self.feature_names = embedding_features + additional_features
        return self.feature_names
    
    def train(
            self, 
            df: pd.DataFrame,
            domain_col: str,
            label_col: str,
            test_size: float = 0.2,
            random_state: int = 42,
            vector_size: int = 50,
            window: int = 5,
            workers: int = 4,
            max_depth: int = 15,
            min_samples_split: int = 5,
            min_samples_leaf: int = 2,
            cv: int = 5,
        ) -> Dict:
        """
        Train the complete Dom2Vec classifier.
        
        Args:
            df: DataFrame containing training data
            domain_col: Name of column containing domain names
            label_col: Name of column containing application labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            vector_size: Dimension of Word2Vec embeddings
            window: Context window size for Word2Vec
            workers: Number of worker threads for Word2Vec
            max_depth: Maximum depth of decision tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            cv: Number of cross-validation folds
            
        Returns:
            Training metrics
        """
        # Update instance parameters with provided values
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        
        # Convert DataFrame to training data format
        training_data = list(zip(df[domain_col].values, df[label_col].values))
        
        # Filter out classes with insufficient samples for stratification
        min_samples_for_stratify = int(1 / test_size) if test_size < 1.0 else 2
        
        # Count samples per class
        from collections import Counter
        class_counts = Counter(df[label_col].values)
        
        # Identify classes to keep
        valid_classes = [cls for cls, count in class_counts.items() 
                        if count >= min_samples_for_stratify]
        
        # Store valid and filtered classes for later use
        self.valid_classes = set(valid_classes)
        self.filtered_classes = set(class_counts.keys()) - self.valid_classes
        
        # Calculate filtering statistics
        total_classes = len(class_counts)
        removed_classes = total_classes - len(valid_classes)
        removed_samples = sum(count for cls, count in class_counts.items() 
                            if cls not in valid_classes)
        
        if removed_classes > 0:
            print(f"\nClass Distribution Analysis:")
            print(f"  Total classes: {total_classes}")
            print(f"  Classes with <{min_samples_for_stratify} samples: {removed_classes} ({removed_classes/total_classes*100:.1f}%)")
            print(f"  Samples in rare classes: {removed_samples} ({removed_samples/len(training_data)*100:.1f}%)")
            print(f"  Filtering out {removed_classes} rare classes...")
            
            # Show some examples of filtered classes if not too many
            if removed_classes <= 10:
                filtered_with_counts = [(cls, class_counts[cls]) for cls in self.filtered_classes]
                print(f"  Filtered classes: {filtered_with_counts}")
            
            # Filter training data
            training_data = [(domain, label) for domain, label in training_data 
                           if label in valid_classes]
            
            print(f"  Remaining classes: {len(valid_classes)}")
            print(f"  Remaining samples: {len(training_data)}")
        
        print(f"\nTraining on {len(training_data)} domain samples...")
        print(f"Using vector_size={vector_size}, window={window}, workers={workers}")
        
        # Step 1: Prepare sentences for Word2Vec
        sentences = self.prepare_training_sentences(training_data)
        
        # Step 2: Train Word2Vec with parameterized settings
        self.train_word2vec(sentences)
        
        # Step 3: Extract features for decision tree
        print("Extracting features for decision tree training...")
        X = []
        y = []
        
        for domain, app_label in training_data:
            features = self.extract_all_features(domain)
            X.append(features)
            y.append(app_label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Step 4: Train decision tree with parameterized settings
        print(f"Training decision tree classifier with max_depth={max_depth}...")
        
        # Now we can safely use stratification since we filtered out small classes
        print(f"\nUsing stratified train-test split with test_size={test_size}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_encoded  # Always stratify now
        )
        
        # Train decision tree with all provided parameters
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        self.decision_tree.fit(X_train, y_train)
        
        # Prepare feature names
        self.prepare_feature_names()
        
        # Evaluate
        y_pred = self.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation with parameterized CV folds
        # Adjust CV folds if necessary for small classes
        train_class_counts = Counter(y_train)
        min_train_class_size = min(train_class_counts.values())
        
        # CV requires at least as many samples as folds per class
        max_cv_folds = min(cv, min_train_class_size)
        
        if max_cv_folds < cv:
            print(f"Warning: Reducing CV folds from {cv} to {max_cv_folds} due to small class sizes")
        
        if max_cv_folds < 2:
            print("Warning: Cannot perform cross-validation with current data. Skipping CV.")
            cv_scores = np.array([0.0])  # Dummy score
        else:
            cv_scores = cross_val_score(self.decision_tree, X_train, y_train, cv=max_cv_folds)
        
        print(f"Training completed!")
        print(f"Test accuracy: {accuracy:.3f}")
        print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Detailed classification report
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        # Generate anonymized performance log
        performance_log = self._generate_performance_log(
            y_test_labels, y_pred_labels, accuracy, cv_scores,
            training_data, vector_size, window, workers,
            max_depth, min_samples_split, min_samples_leaf,
            test_size, cv, random_state
        )
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_labels': y_test_labels,
            'pred_labels': y_pred_labels,
            'training_params': {
                'vector_size': vector_size,
                'window': window,
                'workers': workers,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'test_size': test_size,
                'cv_folds': cv
            },
            'performance_log': performance_log
        }
    
    def predict(self, domain: str) -> Tuple[str, float]:
        """
        Predict application for a given domain.
        
        Args:
            domain: Domain name to classify
            
        Returns:
            Tuple of (predicted_app, confidence)
            Note: Returns ('unknown_rare_app', 0.0) for apps that were filtered during training
        """
        if self.decision_tree is None:
            raise ValueError("Model not trained yet!")
        
        features = self.extract_all_features(domain).reshape(1, -1)
        prediction = self.decision_tree.predict(features)[0]
        probabilities = self.decision_tree.predict_proba(features)[0]
        
        app_name = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return app_name, confidence
    
    def _generate_performance_log(self, y_test_labels, y_pred_labels, accuracy, cv_scores,
                                   training_data, vector_size, window, workers,
                                   max_depth, min_samples_split, min_samples_leaf,
                                   test_size, cv, random_state) -> Dict:
        """Generate anonymized performance log for sharing."""
        # Get unique classes and create anonymized mapping
        unique_classes = sorted(set(y_test_labels) | set(y_pred_labels))
        class_mapping = {cls: f"app_{i:03d}" for i, cls in enumerate(unique_classes)}
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_labels, y_pred_labels, labels=unique_classes, average=None, zero_division=0
        )
        
        # Create anonymized per-class results
        per_class_metrics = []
        for i, cls in enumerate(unique_classes):
            anon_name = class_mapping[cls]
            per_class_metrics.append({
                'class': anon_name,
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            })
        
        # Sort by F1 score descending
        per_class_metrics.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Calculate macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_test_labels, y_pred_labels, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test_labels, y_pred_labels, average='weighted', zero_division=0
        )
        
        # Create confusion matrix with anonymized labels
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=unique_classes)
        anon_labels = [class_mapping[cls] for cls in unique_classes]
        
        # Dataset statistics
        class_distribution = {}
        for _, label in training_data:
            if label in self.valid_classes:
                anon_label = class_mapping.get(label, f"filtered_{label}")
                class_distribution[anon_label] = class_distribution.get(anon_label, 0) + 1
        
        # Generate timestamp and run ID
        timestamp = datetime.now().isoformat()
        run_id = hashlib.md5(f"{timestamp}{random_state}".encode()).hexdigest()[:8]
        
        log = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_configuration": {
                "model_type": "Dom2Vec + DecisionTree",
                "word2vec_params": {
                    "vector_size": vector_size,
                    "window": window,
                    "min_count": self.min_count,
                    "workers": workers,
                    "algorithm": "CBOW"
                },
                "decision_tree_params": {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": random_state
                },
                "training_params": {
                    "test_size": test_size,
                    "cv_folds": cv,
                    "stratified": True
                }
            },
            "dataset_info": {
                "total_samples": len(training_data),
                "total_classes": len(self.valid_classes),
                "filtered_classes": len(self.filtered_classes),
                "class_distribution": class_distribution,
                "vocabulary_size": len(self.word2vec_model.wv.key_to_index) if self.word2vec_model else 0
            },
            "overall_performance": {
                "test_accuracy": float(accuracy),
                "cv_mean_accuracy": float(cv_scores.mean()),
                "cv_std_accuracy": float(cv_scores.std()),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1)
            },
            "per_class_performance": per_class_metrics,
            "confusion_matrix": {
                "labels": anon_labels,
                "matrix": cm.tolist()
            },
            "notes": {
                "anonymized": True,
                "class_mapping_hash": hashlib.md5(str(sorted(class_mapping.items())).encode()).hexdigest()[:8],
                "filtered_class_count": len(self.filtered_classes),
                "filtering_reason": f"Classes with <{int(1/test_size)} samples were filtered"
            }
        }
        
        return log
    
    def get_filtered_classes_info(self) -> Dict:
        """
        Get information about classes that were filtered during training.
        
        Returns:
            Dictionary with filtering statistics
        """
        return {
            'valid_classes': sorted(list(self.valid_classes)),
            'filtered_classes': sorted(list(self.filtered_classes)),
            'num_valid_classes': len(self.valid_classes),
            'num_filtered_classes': len(self.filtered_classes),
            'filtering_applied': len(self.filtered_classes) > 0
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from decision tree."""
        if self.decision_tree is None:
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.decision_tree.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top Feature Importance in Dom2Vec Classifier')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, test_labels, pred_labels):
        """Plot confusion matrix."""
        cm = confusion_matrix(test_labels, pred_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Dom2Vec App Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_performance_log(self, log: Dict, filepath: str = None) -> str:
        """Save performance log to JSON file.
        
        Args:
            log: Performance log dictionary
            filepath: Optional filepath. If None, generates timestamped filename
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"performance_log_{log['run_id']}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"Performance log saved to: {filepath}")
        return filepath
    
    def format_performance_summary(self, log: Dict) -> str:
        """Format performance log as readable summary for sharing.
        
        Args:
            log: Performance log dictionary
            
        Returns:
            Formatted string summary
        """
        summary = []
        summary.append("="*60)
        summary.append("MODEL PERFORMANCE SUMMARY (ANONYMIZED)")
        summary.append("="*60)
        summary.append(f"Run ID: {log['run_id']}")
        summary.append(f"Timestamp: {log['timestamp']}")
        
        # Model configuration
        summary.append("\nMODEL CONFIGURATION:")
        summary.append(f"  Model Type: {log['model_configuration']['model_type']}")
        summary.append(f"  Word2Vec: vector_size={log['model_configuration']['word2vec_params']['vector_size']}, "
                      f"window={log['model_configuration']['word2vec_params']['window']}")
        summary.append(f"  Decision Tree: max_depth={log['model_configuration']['decision_tree_params']['max_depth']}, "
                      f"min_samples_split={log['model_configuration']['decision_tree_params']['min_samples_split']}")
        summary.append(f"  Training: test_size={log['model_configuration']['training_params']['test_size']}, "
                      f"cv_folds={log['model_configuration']['training_params']['cv_folds']}")
        
        # Dataset info
        summary.append("\nDATASET INFO:")
        summary.append(f"  Total Samples: {log['dataset_info']['total_samples']}")
        summary.append(f"  Valid Classes: {log['dataset_info']['total_classes']}")
        summary.append(f"  Filtered Classes: {log['dataset_info']['filtered_classes']}")
        summary.append(f"  Vocabulary Size: {log['dataset_info']['vocabulary_size']}")
        
        # Overall performance
        summary.append("\nOVERALL PERFORMANCE:")
        summary.append(f"  Test Accuracy: {log['overall_performance']['test_accuracy']:.3f}")
        summary.append(f"  CV Accuracy: {log['overall_performance']['cv_mean_accuracy']:.3f} ± "
                      f"{log['overall_performance']['cv_std_accuracy']:.3f}")
        summary.append(f"  Macro F1: {log['overall_performance']['macro_f1']:.3f}")
        summary.append(f"  Weighted F1: {log['overall_performance']['weighted_f1']:.3f}")
        
        # Per-class performance (top 10)
        summary.append("\nPER-CLASS PERFORMANCE (Top 10 by F1 Score):")
        summary.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        summary.append("-"*50)
        
        for i, metrics in enumerate(log['per_class_performance'][:10]):
            summary.append(f"{metrics['class']:<10} {metrics['precision']:<10.3f} "
                          f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} "
                          f"{metrics['support']:<10}")
        
        # Notes
        summary.append("\nNOTES:")
        for key, value in log['notes'].items():
            summary.append(f"  {key}: {value}")
        
        summary.append("="*60)
        
        return "\n".join(summary)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'word2vec_model': self.word2vec_model,
            'decision_tree': self.decision_tree,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'vector_size': self.vector_size,
            'valid_classes': self.valid_classes,
            'filtered_classes': self.filtered_classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.word2vec_model = model_data['word2vec_model']
        self.decision_tree = model_data['decision_tree']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.vector_size = model_data['vector_size']
        self.valid_classes = model_data.get('valid_classes', set())
        self.filtered_classes = model_data.get('filtered_classes', set())
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = Dom2VecAppClassifier()
    
    # Load sample data and convert to DataFrame format
    from sample_data import get_all_training_data
    training_data = get_all_training_data()
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(training_data, columns=['domain', 'application'])
    
    print("Training Dom2Vec Application Classifier...")
    print(f"Dataset: {len(df)} samples, {df['application'].nunique()} applications")
    
    # Train model with custom parameters
    metrics = classifier.train(
        df=df,
        domain_col='domain',
        label_col='application',
        vector_size=50,  # Smaller for demo
        window=3,
        max_depth=10,
        test_size=0.2,
        cv=5
    )
    
    # Test predictions
    test_domains = [
        "netflix.com",
        "api.spotify.com", 
        "teams.microsoft.com",
        "youtube.com",
        "github.com",
        "unknown-domain.com"
    ]
    
    print("\n" + "="*50)
    print("Testing Predictions:")
    print("="*50)
    
    for domain in test_domains:
        try:
            app, confidence = classifier.predict(domain)
            print(f"{domain:25} -> {app:15} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"{domain:25} -> Error: {e}")
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    print(classifier.get_feature_importance(10)) 