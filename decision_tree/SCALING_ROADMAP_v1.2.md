# Million-Domain Scaling Roadmap v1.2

**Current Status**: Production-ready for 10K-100K domains  
**Target**: Million-domain scale with <10ms inference  
**Timeline**: 3-6 months implementation  

## Current Bottlenecks Analysis

### 🚨 **Critical Scaling Issues**

1. **Algorithm Scalability**
   - Decision Tree: O(n log n) training complexity
   - Single-machine processing only
   - Memory grows linearly with dataset size

2. **Feature Engineering Bottlenecks**
   - 107D feature vector too large for million samples
   - Word2Vec vocabulary explosion (417 → 50K+ tokens)
   - Simple averaging inadequate for large vocabularies

3. **Infrastructure Limitations**
   - No distributed training capability
   - No streaming inference pipeline
   - No model versioning/deployment system

## Implementation Roadmap

### **Phase 1: Core Algorithm Replacement (Month 1)**

#### Replace Decision Tree with LightGBM
```python
# Current: Single Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=15)

# Recommended: LightGBM for scalability
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    objective='multiclass',
    n_jobs=-1  # Use all cores
)
```

**Benefits:**
- Handles 1M+ samples efficiently
- Built-in feature selection
- Distributed training support
- 10x faster than Decision Trees

#### Implement Feature Reduction
```python
# Reduce from 107D to 25D maximum
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Method 1: PCA on embeddings
pca = PCA(n_components=20)
embedding_features_reduced = pca.fit_transform(embedding_features)

# Method 2: Feature selection
selector = SelectKBest(k=5)
structural_features_selected = selector.fit_transform(structural_features)

# Final: 20D + 5D = 25D feature vector
```

### **Phase 2: Streaming Architecture (Month 2)**

#### Implement Batch Training Pipeline
```python
# Stream processing for large datasets
def train_in_batches(dataset_path, batch_size=10000):
    model = lgb.LGBMClassifier()
    
    for batch in read_batches(dataset_path, batch_size):
        X_batch, y_batch = prepare_features(batch)
        
        # Incremental training
        if model.booster_ is None:
            model.fit(X_batch, y_batch)
        else:
            # Continue training on new batch
            model.fit(X_batch, y_batch, init_model=model.booster_)
    
    return model
```

#### Add Distributed Processing
```python
# Use Dask for distributed feature processing
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split

# Parallel feature extraction
def extract_features_distributed(domains_df):
    # Shard by hash for parallel processing
    return domains_df.map_partitions(extract_domain_features)
```

### **Phase 3: Advanced Embeddings (Month 3)**

#### Replace Word2Vec with Scalable Alternatives
```python
# Option 1: FastText (handles out-of-vocabulary better)
from gensim.models import FastText
model = FastText(
    sentences=domain_tokens,
    vector_size=50,  # Reduced from 100
    window=3,        # Reduced from 5
    min_count=5,     # Increased threshold
    workers=8
)

# Option 2: Bloom Filter Embeddings for rare tokens
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=1024, input_type='string')
rare_token_features = hasher.transform(rare_tokens)
```

#### Implement Attention-based Pooling
```python
# Replace simple averaging with weighted pooling
def attention_pooled_embedding(token_embeddings, token_frequencies):
    # Weight by inverse frequency (TF-IDF style)
    weights = 1.0 / (1.0 + token_frequencies)
    weights = weights / weights.sum()
    
    return np.average(token_embeddings, axis=0, weights=weights)
```

### **Phase 4: Production Infrastructure (Month 4)**

#### Model Serving Pipeline
```python
# High-throughput serving with FastAPI
from fastapi import FastAPI
import asyncio

app = FastAPI()

# Load model once at startup
model = load_model("production_model_v1.2.pkl")
feature_extractor = DomainFeatureExtractor()

@app.post("/predict")
async def predict_domain(domain: str):
    # Sub-10ms inference target
    features = feature_extractor.extract(domain)
    prediction = model.predict_proba([features])[0]
    
    return {
        "domain": domain,
        "predicted_app": model.classes_[prediction.argmax()],
        "confidence": float(prediction.max()),
        "inference_time_ms": "< 10ms"
    }
```

#### Caching and Optimization
```python
# Redis cache for frequent domains
import redis
cache = redis.Redis()

def predict_with_cache(domain):
    # Check cache first
    cached = cache.get(f"domain:{domain}")
    if cached:
        return json.loads(cached)
    
    # Compute prediction
    result = model.predict(domain)
    
    # Cache for 1 hour
    cache.setex(f"domain:{domain}", 3600, json.dumps(result))
    return result
```

## Performance Targets

| Metric | Current (8K domains) | Target (1M domains) | Implementation |
|--------|---------------------|---------------------|----------------|
| **Training Time** | 0.1 seconds | <1 hour | LightGBM + batching |
| **Inference Latency** | ~50ms | <10ms p99 | FastAPI + caching |
| **Memory Usage** | ~10MB | <8GB | Feature reduction |
| **Accuracy** | 76.3% | >75% | Maintain performance |
| **Throughput** | ~10 qps | >1000 qps | Horizontal scaling |

## Risk Assessment & Mitigation

### **High Risk: Model Quality Degradation**
- **Risk**: Accuracy drops with feature reduction
- **Mitigation**: A/B testing, gradual rollout
- **Fallback**: Keep decision tree for comparison

### **Medium Risk: Infrastructure Complexity**
- **Risk**: Distributed systems introduce failure points
- **Mitigation**: Comprehensive monitoring, circuit breakers
- **Fallback**: Graceful degradation to single-machine mode

### **Low Risk: Integration Challenges**
- **Risk**: Existing systems integration
- **Mitigation**: Backward-compatible APIs
- **Fallback**: Dual deployment during transition

## Implementation Priority

### **Immediate (Week 1-2)**
1. Implement LightGBM replacement
2. Add feature reduction pipeline
3. Benchmark performance on current dataset

### **Short-term (Month 1)**
1. Streaming batch training
2. Performance optimization
3. Basic caching implementation

### **Medium-term (Month 2-3)**
1. Distributed processing
2. Advanced embedding strategies
3. Production serving infrastructure

### **Long-term (Month 4-6)**
1. Full production deployment
2. Online learning capabilities
3. Monitoring and alerting systems

## Success Criteria

**Phase 1 Success:** Train on 100K domains in <10 minutes with >70% accuracy
**Phase 2 Success:** Process 1M domains in <1 hour with distributed pipeline  
**Phase 3 Success:** Serve predictions at <10ms latency with >1000 qps
**Phase 4 Success:** Full production deployment with monitoring

## Conclusion

The current model (76.3% accuracy) provides an excellent foundation, but **requires significant architectural changes** for million-domain scale. The roadmap above addresses all major bottlenecks while maintaining prediction quality.

**Estimated timeline: 3-6 months** for full million-domain readiness.

---

**Next Steps:**
1. Begin Phase 1 implementation (LightGBM + feature reduction)
2. Validate performance on 100K synthetic domains
3. Proceed through phases with continuous validation