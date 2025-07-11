# Dom2Vec Application Classifier

A machine learning system that classifies applications from SNI domain names using Word2Vec embeddings and Decision Trees, adapted from the Dom2Vec approach for DGA detection.

## Overview

This implementation adapts the Dom2Vec technique (originally designed for DGA detection) to classify applications from domain names seen in network traffic. It combines:

- **Domain Processing**: Intelligent tokenization of domain names into meaningful components
- **Word2Vec Embeddings**: Vector representations that capture semantic relationships between domain tokens
- **Decision Tree Classification**: Interpretable machine learning for application prediction
- **Additional Features**: Domain-specific patterns and characteristics

## Key Features

- ✅ **Real-world Applications**: Trained on domains from Netflix, Spotify, YouTube, Microsoft Teams, Zoom, GitHub, etc.
- ✅ **Intelligent Tokenization**: Splits domains like `api.spotify.com` → `['api', 'spotify', 'com']`
- ✅ **Semantic Understanding**: Uses Word2Vec to understand relationships between domain components
- ✅ **Interpretable Results**: Decision trees provide clear decision paths
- ✅ **Pattern Recognition**: Detects API endpoints, CDN services, company-specific patterns
- ✅ **Extensible**: Easy to add new applications and retrain

## File Structure

```
decision_tree/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── domain_processor.py            # Domain tokenization and feature extraction
├── sample_data.py                 # Training data with real-world domain examples
├── dom2vec_app_classifier.py      # Main classifier implementation
├── train_model.py                 # Training script with configurable parameters
├── test_classifier.py             # Testing and interactive demo script
└── Dom2Vec DGA Domains Analysis.pdf  # Original research paper
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Domain Processing

```bash
python domain_processor.py
```

This will show how domains get tokenized:
```
Domain: api.spotify.com
Tokens: ['api', 'spotify', 'com']
Key features: domain_length=15, has_api=True, has_cdn=False
```

### 3. Train the Model

```bash
# Basic training
python train_model.py

# With plots and verbose output
python train_model.py --plot --verbose

# Custom parameters
python train_model.py --vector-size 100 --max-depth 20 --plot
```

### 4. Test the Trained Model

```bash
# Test with default domains
python test_classifier.py

# Interactive mode
python test_classifier.py --interactive

# Test specific domains
python test_classifier.py --domains netflix.com api.spotify.com teams.microsoft.com

# Analyze how a domain gets processed
python test_classifier.py --analyze "audio-ak-spotify-com.akamaized.net"
```

## Usage Examples

### Training with Custom Parameters

```bash
python train_model.py \
    --vector-size 150 \
    --window 7 \
    --max-depth 20 \
    --min-samples-split 10 \
    --plot \
    --verbose \
    --save-model my_model.pkl
```

### Interactive Testing

```bash
python test_classifier.py --interactive
```

```
Domain: nflxvideo.net
Results for: nflxvideo.net
  Predicted App: netflix
  Confidence:    0.892
  Tokens:        ['nflx', 'video', 'net']
  Domain Length: 13
  Subdomains:    0
  Has API:       False
  Has CDN:       False
```

### Programmatic Usage

```python
from dom2vec_app_classifier import Dom2VecAppClassifier

# Load trained model
classifier = Dom2VecAppClassifier()
classifier.load_model('dom2vec_model.pkl')

# Predict application
app, confidence = classifier.predict('api.spotify.com')
print(f"Application: {app}, Confidence: {confidence:.3f}")
# Output: Application: spotify, Confidence: 0.945
```

## How It Works

### 1. Domain Tokenization

The system intelligently splits domain names into meaningful tokens:

```python
# Examples of tokenization
"netflix.com" → ['netflix', 'com']
"api.spotify.com" → ['api', 'spotify', 'com'] 
"nflxvideo.net" → ['nflx', 'video', 'net']
"teams.microsoft.com" → ['teams', 'microsoft', 'com']
"fonts.googleapis.com" → ['fonts', 'google', 'apis', 'com']
```

### 2. Word2Vec Training

Creates vector embeddings where similar tokens have similar vectors:
- `'spotify'` and `'music'` will be close in vector space
- `'api'` and `'rest'` will be related
- `'google'` and `'youtube'` will show company relationships

### 3. Feature Combination

Combines embeddings with domain-specific features:
- **Embedding features**: 100-dimensional Word2Vec vectors (averaged across tokens)
- **Structural features**: Domain length, subdomain count, character patterns
- **Pattern features**: Has API keywords, CDN patterns, company indicators

### 4. Decision Tree Classification

Uses interpretable decision trees that can show reasoning like:
```
If embed_42 <= 0.23:
  └── If has_spotify_pattern == True: → Spotify
  └── If domain_length <= 12: → Netflix
Else:
  └── If has_api_pattern == True: → Google Services
```

## Training Data

The model is trained on real-world domains from popular applications:

| Application | Example Domains | Count |
|-------------|----------------|-------|
| Netflix | netflix.com, nflxvideo.net, nflximg.net | 13 |
| Spotify | spotify.com, scdn.co, api.spotify.com | 13 |
| YouTube | youtube.com, googlevideo.com, ytimg.com | 13 |
| Microsoft Teams | teams.microsoft.com, teams.live.com | 12 |
| Zoom | zoom.us, zmcdn.com, api.zoom.us | 13 |
| Google Services | google.com, googleapis.com, gstatic.com | 18 |
| Facebook/Meta | facebook.com, instagram.com, whatsapp.com | 15 |
| Amazon/AWS | amazon.com, amazonaws.com, cloudfront.net | 14 |
| Apple | apple.com, icloud.com, mzstatic.com | 13 |
| GitHub | github.com, githubusercontent.com | 10 |
| Slack | slack.com, slack-edge.com, api.slack.com | 10 |

## Performance

Expected performance metrics:
- **Accuracy**: ~85-95% on test data
- **Training Time**: ~10-30 seconds on standard hardware
- **Prediction Time**: <1ms per domain
- **Model Size**: ~1-5MB (depending on vocabulary size)

## Key Differences from Original Dom2Vec

| Aspect | Original Dom2Vec | This Implementation |
|--------|------------------|-------------------|
| **Purpose** | DGA vs legitimate detection | Application classification |
| **Output** | Binary classification | Multi-class (11 applications) |
| **Features** | Embeddings + reputation scores | Embeddings + domain patterns |
| **Training Data** | Malicious vs benign domains | Application-labeled domains |
| **Patterns** | Malware family signatures | Company/service signatures |

## Extending the Model

### Adding New Applications

1. **Add training data** in `sample_data.py`:
```python
APPLICATION_DOMAINS['new_app'] = [
    'newapp.com',
    'api.newapp.com',
    'cdn.newapp.com'
]
```

2. **Add pattern detection** in `domain_processor.py`:
```python
'has_newapp_pattern': any(word in tokens for word in ['newapp', 'new']),
```

3. **Retrain the model**:
```bash
python train_model.py --save-model updated_model.pkl
```

### Tuning Parameters

- **Vector Size**: Larger embeddings (100-300) for more complex relationships
- **Window Size**: Larger windows (5-10) for more context
- **Tree Depth**: Deeper trees (15-25) for more complex decision boundaries
- **Min Samples**: Higher values (10-20) to prevent overfitting

## Research Background

This implementation is based on the Dom2Vec paper:
> "Dom2Vec - Detecting DGA Domains Through Word Embeddings and AI/ML-Driven Lexicographic Analysis"
> 
> *2023 19th International Conference on Network and Service Management (CNSM)*

The key innovation was adapting the domain embedding approach from malware detection to application classification, making it suitable for network traffic analysis and application identification.

## License

This implementation is for educational and research purposes. Please refer to the original Dom2Vec paper for research citations.

## Contributing

To contribute:
1. Add more real-world domain examples to `sample_data.py`
2. Improve tokenization logic in `domain_processor.py`
3. Add new features for better classification
4. Test with your own network traffic data

## Troubleshooting

### Common Issues

1. **NLTK Download Error**: Run `python -c "import nltk; nltk.download('words')"`
2. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
3. **Model File Not Found**: Train a model first: `python train_model.py`
4. **Low Accuracy**: Try increasing `--vector-size` or adding more training data

### Performance Tips

- Use larger vector sizes (100-200) for better accuracy
- Add more diverse training domains for each application
- Tune decision tree parameters based on your data
- Consider ensemble methods for production use 