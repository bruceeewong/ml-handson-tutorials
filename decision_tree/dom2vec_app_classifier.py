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

from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        print(f"Training on {len(training_data)} domain samples...")
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
        
        # Check class distribution for stratification
        from collections import Counter
        class_counts = Counter(y_encoded)
        min_class_size = min(class_counts.values())
        
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Minimum class size: {min_class_size}")
        
        # Determine if stratification is possible
        min_samples_needed = int(1 / test_size) if test_size < 1.0 else 2
        use_stratify = min_class_size >= min_samples_needed
        
        if not use_stratify:
            print(f"Warning: Some classes have only {min_class_size} samples.")
            print(f"Cannot use stratified split (need ≥{min_samples_needed} per class).")
            print("Using random split instead.")
        
        # Split data with conditional stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_encoded if use_stratify else None
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
            }
        }
    
    def predict(self, domain: str) -> Tuple[str, float]:
        """
        Predict application for a given domain.
        
        Args:
            domain: Domain name to classify
            
        Returns:
            Tuple of (predicted_app, confidence)
        """
        if self.decision_tree is None:
            raise ValueError("Model not trained yet!")
        
        features = self.extract_all_features(domain).reshape(1, -1)
        prediction = self.decision_tree.predict(features)[0]
        probabilities = self.decision_tree.predict_proba(features)[0]
        
        app_name = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return app_name, confidence
    
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
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'word2vec_model': self.word2vec_model,
            'decision_tree': self.decision_tree,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'vector_size': self.vector_size
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