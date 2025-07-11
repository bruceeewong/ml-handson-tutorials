# Dom2Vec Application Classifier - Synthetic Data Generation Strategy

**Document Version**: 1.1  
**Date**: January 11, 2025  
**Author**: Technical Analysis Team  
**Status**: Synthetic Data Expansion Phase  
**Previous Version**: [DESIGN_ANALYSIS_v1.0.md](DESIGN_ANALYSIS_v1.0.md)

## Executive Summary

This document outlines the strategy and implementation plan for expanding the Dom2Vec training dataset from 144 real domains to 10,000+ synthetic domains. The primary goal is to address the curse of dimensionality issue (107 features vs 144 samples) identified in v1.0, which is the root cause of the current 37.9% accuracy.

## Table of Contents

1. [Current Dataset Analysis](#current-dataset-analysis)
2. [Synthetic Data Generation Strategy](#synthetic-data-generation-strategy)
3. [Implementation Plan](#implementation-plan)
4. [Quality Assurance Framework](#quality-assurance-framework)
5. [Expected Outcomes](#expected-outcomes)
6. [Risk Mitigation](#risk-mitigation)
7. [Next Steps](#next-steps)

## Current Dataset Analysis

### Dataset Statistics

**Current State**:
```
Total domains: 144
Applications: 11
Domains per app: 10-18 (avg: 13.1)
Unique tokens: 161
Token frequency: 58% appear only once
```

### Domain Pattern Analysis

From analyzing the existing 144 domains, we've identified key patterns:

#### 1. Netflix Domains (13 samples)
```
Primary patterns:
- Main domain: netflix.com
- CDN: nflxvideo.net, nflximg.net, nflxext.com
- API: api.netflix.com, api-global.netflix.com
- Regional: netflix.ca, netflix.co.uk
```

#### 2. Spotify Domains (13 samples)
```
Primary patterns:
- Main: spotify.com, open.spotify.com
- CDN: scdn.co, spotifycdn.com
- API: api.spotify.com, accounts.spotify.com
- Services: audio-fa.scdn.co, i.scdn.co
```

#### 3. Common Domain Patterns Across Applications

| Pattern Type | Example | Frequency |
|-------------|---------|-----------|
| API endpoints | api.{service}.com | 72% |
| CDN/Static | cdn.{service}.com, {service}cdn.com | 65% |
| Regional | {service}.{country} | 45% |
| Subservices | {function}.{service}.com | 83% |
| Short domains | {abbrev}.co | 35% |

### Token Distribution Analysis

**Top 20 Most Common Tokens**:
```python
token_frequency = {
    'com': 118,
    'net': 24,
    'api': 19,
    'cdn': 15,
    'cloud': 12,
    'services': 11,
    'content': 9,
    'static': 8,
    'media': 7,
    'app': 7,
    # ... continues
}
```

## Synthetic Data Generation Strategy

### Phase 1: Template-Based Generation (Target: 1,000 domains/app)

#### 1.1 Core Templates

```python
DOMAIN_TEMPLATES = {
    'api_endpoints': [
        'api.{service}.{tld}',
        'api-{region}.{service}.{tld}',
        '{service}-api.{tld}',
        'api.{subdomain}.{service}.{tld}',
        '{version}.api.{service}.{tld}'
    ],
    'cdn_patterns': [
        'cdn.{service}.{tld}',
        '{service}cdn.{tld}',
        'static.{service}.{tld}',
        'assets.{service}.{tld}',
        '{region}-cdn.{service}.{tld}',
        'cdn{number}.{service}.{tld}'
    ],
    'regional': [
        '{service}.{country_code}',
        '{region}.{service}.{tld}',
        '{service}-{region}.{tld}'
    ],
    'services': [
        '{function}.{service}.{tld}',
        '{service}-{function}.{tld}',
        '{function}-{service}.{tld}'
    ],
    'mobile': [
        'm.{service}.{tld}',
        'mobile.{service}.{tld}',
        'app.{service}.{tld}',
        '{service}-mobile.{tld}'
    ]
}
```

#### 1.2 Application-Specific Variations

```python
SERVICE_VARIATIONS = {
    'netflix': {
        'base': ['netflix', 'nflx'],
        'functions': ['watch', 'stream', 'video', 'movies', 'shows'],
        'cdn_terms': ['nflxvideo', 'nflximg', 'nflxext'],
        'regions': ['us', 'eu', 'asia', 'latam']
    },
    'spotify': {
        'base': ['spotify', 'sptfy', 'spot'],
        'functions': ['music', 'audio', 'podcast', 'play', 'stream'],
        'cdn_terms': ['scdn', 'spotifycdn', 'spoticdn'],
        'regions': ['us', 'eu', 'uk', 'br', 'jp']
    },
    # ... continues for all 11 applications
}
```

### Phase 2: Intelligent Mutations (Target: +500 domains/app)

#### 2.1 Linguistic Variations
```python
MUTATIONS = {
    'abbreviations': {
        'service': 'svc', 'srv',
        'content': 'cnt', 'cont',
        'delivery': 'del', 'dlv',
        'network': 'net', 'nw'
    },
    'synonyms': {
        'api': ['rest', 'graphql', 'endpoint', 'gateway'],
        'cdn': ['edge', 'cache', 'static', 'assets'],
        'media': ['content', 'stream', 'video', 'audio']
    },
    'compounds': {
        'patterns': ['{word1}-{word2}', '{word1}{word2}', '{w1}.{w2}']
    }
}
```

#### 2.2 Realistic Noise Injection
```python
REALISTIC_VARIANTS = {
    'versioning': ['v1', 'v2', 'v3', 'beta', 'alpha', 'staging'],
    'environments': ['prod', 'production', 'dev', 'test', 'qa'],
    'load_balancing': ['lb1', 'lb2', 'node1', 'node2', 'cluster1'],
    'dates': ['2023', '2024', '2025'],
    'random_ids': ['xyz123', 'abc456', 'tmp789']
}
```

### Phase 3: Cross-Application Patterns (Target: +200 domains/app)

Generate domains that could plausibly belong to multiple applications to test disambiguation:

```python
AMBIGUOUS_PATTERNS = [
    'streaming.{tld}',          # Netflix? Spotify? YouTube?
    'media-cdn.{tld}',          # Could be any media service
    'api-gateway.{tld}',        # Generic API pattern
    'cloud-services.{tld}',     # AWS? Azure? GCP?
    'user-content.{tld}'        # Facebook? GitHub? Google?
]
```

## Implementation Plan

### Step 1: Data Generator Class

```python
class DomainSynthesizer:
    def __init__(self, real_domains, templates, mutations):
        self.real_domains = real_domains
        self.templates = templates
        self.mutations = mutations
        self.generated_domains = set()
        
    def generate_from_template(self, app_name, template, count=100):
        """Generate domains using template substitution"""
        domains = []
        variations = SERVICE_VARIATIONS[app_name]
        
        for _ in range(count):
            domain = template
            # Substitute placeholders with variations
            domain = self._substitute_placeholders(domain, variations)
            
            # Ensure uniqueness
            if domain not in self.generated_domains:
                domains.append(domain)
                self.generated_domains.add(domain)
                
        return domains
    
    def apply_mutations(self, domain):
        """Apply realistic mutations to existing domains"""
        mutated = []
        
        # Abbreviation mutations
        for word, abbrevs in self.mutations['abbreviations'].items():
            if word in domain:
                for abbrev in abbrevs:
                    mutated.append(domain.replace(word, abbrev))
        
        # Synonym replacements
        for word, synonyms in self.mutations['synonyms'].items():
            if word in domain:
                for synonym in synonyms:
                    mutated.append(domain.replace(word, synonym))
                    
        return mutated
    
    def validate_domain(self, domain):
        """Ensure generated domain is realistic"""
        # Check length constraints
        if len(domain) < 5 or len(domain) > 63:
            return False
            
        # Check valid characters
        if not re.match(r'^[a-z0-9.-]+$', domain):
            return False
            
        # Check TLD validity
        if not any(domain.endswith(tld) for tld in VALID_TLDS):
            return False
            
        return True
```

### Step 2: Generation Pipeline

```python
def generate_synthetic_dataset(target_per_app=1000):
    synthesizer = DomainSynthesizer(
        real_domains=CURRENT_DATASET,
        templates=DOMAIN_TEMPLATES,
        mutations=MUTATIONS
    )
    
    synthetic_data = {}
    
    for app_name in APPLICATION_NAMES:
        app_domains = []
        
        # Phase 1: Template-based generation (70%)
        for template_type, templates in DOMAIN_TEMPLATES.items():
            for template in templates:
                domains = synthesizer.generate_from_template(
                    app_name, template, count=int(target_per_app * 0.7 / len(templates))
                )
                app_domains.extend(domains)
        
        # Phase 2: Mutation-based generation (20%)
        real_app_domains = CURRENT_DATASET[app_name]
        for real_domain in real_app_domains:
            mutated = synthesizer.apply_mutations(real_domain)
            app_domains.extend(mutated[:int(target_per_app * 0.2 / len(real_app_domains))])
        
        # Phase 3: Ambiguous patterns (10%)
        ambiguous = synthesizer.generate_ambiguous_domains(
            app_name, count=int(target_per_app * 0.1)
        )
        app_domains.extend(ambiguous)
        
        # Validate and store
        synthetic_data[app_name] = [
            domain for domain in app_domains 
            if synthesizer.validate_domain(domain)
        ][:target_per_app]
    
    return synthetic_data
```

### Step 3: Integration with Training Pipeline

```python
def augment_training_data(original_data, synthetic_data, mix_ratio=0.8):
    """
    Mix original and synthetic data
    mix_ratio: proportion of synthetic data in final dataset
    """
    augmented_data = {}
    
    for app_name in APPLICATION_NAMES:
        original = original_data[app_name]
        synthetic = synthetic_data[app_name]
        
        # Always include all original data
        combined = original.copy()
        
        # Add synthetic data up to mix_ratio
        n_synthetic = int(len(synthetic) * mix_ratio)
        combined.extend(random.sample(synthetic, n_synthetic))
        
        augmented_data[app_name] = combined
    
    return augmented_data
```

## Quality Assurance Framework

### 1. Domain Realism Metrics

```python
class DomainQualityChecker:
    def __init__(self, real_domains):
        self.real_domains = real_domains
        self._build_statistics()
    
    def _build_statistics(self):
        """Analyze real domains for patterns"""
        self.stats = {
            'avg_length': np.mean([len(d) for d in self.real_domains]),
            'std_length': np.std([len(d) for d in self.real_domains]),
            'token_frequency': self._calculate_token_frequency(),
            'subdomain_levels': self._analyze_subdomain_levels(),
            'tld_distribution': self._analyze_tlds()
        }
    
    def score_synthetic_domain(self, domain):
        """Score 0-1 for how realistic a synthetic domain is"""
        scores = []
        
        # Length score
        length_z = abs(len(domain) - self.stats['avg_length']) / self.stats['std_length']
        scores.append(1 / (1 + length_z))
        
        # Token familiarity score
        tokens = domain.split('.')
        token_score = sum(self.stats['token_frequency'].get(t, 0) for t in tokens) / len(tokens)
        scores.append(min(token_score, 1.0))
        
        # Structure score
        subdomain_count = domain.count('.')
        expected = self.stats['subdomain_levels']['mean']
        structure_score = 1 - abs(subdomain_count - expected) / 3
        scores.append(max(structure_score, 0))
        
        return np.mean(scores)
```

### 2. Diversity Metrics

```python
def calculate_diversity_metrics(synthetic_domains):
    """Ensure synthetic data has sufficient diversity"""
    metrics = {
        'unique_tokens': len(set(token for d in synthetic_domains for token in d.split('.'))),
        'unique_patterns': len(set(d.count('.') for d in synthetic_domains)),
        'character_entropy': calculate_entropy(''.join(synthetic_domains)),
        'token_entropy': calculate_entropy([t for d in synthetic_domains for t in d.split('.')]),
        'length_variance': np.var([len(d) for d in synthetic_domains])
    }
    return metrics
```

### 3. Classification Challenge Test

```python
def test_classification_difficulty(synthetic_data):
    """Ensure synthetic data isn't too easy to classify"""
    # Train a simple model on synthetic data only
    simple_model = DecisionTreeClassifier(max_depth=5)
    
    # If accuracy is too high, synthetic data is too simple
    accuracy = cross_val_score(simple_model, X_synthetic, y_synthetic, cv=5).mean()
    
    if accuracy > 0.95:
        logger.warning("Synthetic data may be too simple - consider adding more noise")
    
    return accuracy
```

## Expected Outcomes

### Quantitative Targets

| Metric | Current (144 samples) | Target (10K+ samples) |
|--------|----------------------|----------------------|
| Test Accuracy | 37.9% | 75-85% |
| Cross-Validation Std | ±9.0% | ±2.0% |
| Feature/Sample Ratio | 0.74 | 10.0+ |
| Training Time | 0.09s | <5s |
| Vocabulary Size | 161 | 2,000-5,000 |

### Qualitative Improvements

1. **Better Generalization**: Model learns patterns, not memorizes domains
2. **Robust Features**: Word2Vec embeddings become meaningful with more data
3. **Confident Predictions**: Reduced variance in confidence scores
4. **Handle Edge Cases**: Better performance on unusual domain patterns

## Risk Mitigation

### Risk 1: Synthetic Data Distribution Mismatch
**Mitigation**: 
- Validate against held-out real domains
- Iteratively adjust generation parameters
- Use adversarial validation to detect distribution shift

### Risk 2: Over-Representation of Patterns
**Mitigation**:
- Enforce diversity constraints during generation
- Limit repetitions of similar patterns
- Add controlled noise to break patterns

### Risk 3: Model Overfitting to Synthetic Patterns
**Mitigation**:
- Always include all original domains in training
- Use stratified sampling in train/test split
- Monitor performance on real vs synthetic test sets separately

## Next Steps

### Immediate Actions (Week 1)
1. Implement `DomainSynthesizer` class
2. Generate initial 1,000 synthetic domains per app
3. Validate quality metrics on synthetic data
4. Train model with 50/50 real/synthetic mix

### Short-term Goals (Month 1)
1. Scale to 5,000 domains per application
2. Implement advanced mutation strategies
3. Add cross-application ambiguous patterns
4. Achieve 65%+ accuracy milestone

### Long-term Vision (Months 2-3)
1. Expand to 10,000+ domains per application
2. Implement online learning for new patterns
3. Build domain generation API for continuous updates
4. Prepare for million-domain scale (per v1.0 recommendations)

## Implementation Code Structure

```
decision_tree/
├── DESIGN_ANALYSIS_v1.0.md
├── DESIGN_ANALYSIS_v1.1.md (this document)
├── domain_processor.py (existing)
├── dom2vec_app_classifier.py (existing)
├── synthetic_data/
│   ├── __init__.py
│   ├── domain_synthesizer.py
│   ├── quality_checker.py
│   ├── templates.py
│   └── mutations.py
├── data/
│   ├── original_domains.json
│   ├── synthetic_domains_v1.json
│   └── combined_dataset.json
└── tests/
    ├── test_synthesizer.py
    └── test_quality_metrics.py
```

---

**Document Maintainers**: Technical Analysis Team  
**Review Status**: Ready for Implementation  
**Dependencies**: Requires completion of v1.0 analysis  
**Next Review**: After initial synthetic data generation (Week 1)