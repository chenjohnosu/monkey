# Keyword Extraction Configuration Guide

## Overview
The keyword extraction system now supports multiple methods with SpaCy as the default approach.

## Configuration Options

### Basic Configuration
In your `config.yaml`:

```yaml
keywords:
  method: 'spacy'  # Default method
  max_ngram_size: 2  # Maximum phrase length
```

### Available Methods
1. **SpaCy** (`'spacy'`): 
   - Recommended default
   - Uses SpaCy's advanced NLP capabilities
   - Extracts meaningful noun chunks and named entities
   - Language-aware extraction

2. **TF-IDF** (`'tf-idf'`):
   - Traditional statistical approach
   - Identifies important terms based on frequency
   - Works well for large document collections

3. **RAKE** (`'rake-nltk'`):
   - Rapid Automatic Keyword Extraction
   - Good for extracting meaningful phrases
   - Works best with English text

4. **YAKE** (`'yake'`):
   - Yet Another Keyword Extractor
   - Unsupervised keyword extraction
   - Works across multiple languages

5. **KeyBERT** (`'keybert'`):
   - BERT-based keyword extraction
   - Uses transformer models
   - Advanced semantic understanding

## Method Selection Criteria

### SpaCy (Recommended)
- Best for: Multilingual, semantically rich extraction
- Strengths:
  - Language-aware extraction
  - Named entity recognition
  - Part-of-speech based filtering

### TF-IDF
- Best for: Large document collections
- Strengths:
  - Statistical significance
  - Simple and fast
  - Works across languages

### RAKE
- Best for: Phrase extraction in English
- Strengths:
  - Extracts meaningful multi-word phrases
  - Good for specific domain texts

### YAKE
- Best for: Multilingual keyword extraction
- Strengths:
  - Works across different languages
  - Unsupervised approach
  - Handles various text types

### KeyBERT
- Best for: Semantic keyword extraction
- Strengths:
  - Uses transformer models
  - Captures deep semantic meanings
  - Works well with contextual understanding

## Configuration Example

```yaml
keywords:
  method: 'spacy'  # Primary method
  max_ngram_size: 2  # Control phrase length
  fallback_methods:  # Optional fallback chain
    - 'tf-idf'
    - 'yake'
    - 'rake-nltk'
```

## Performance Considerations

1. **Model Selection**
   - Choose based on document type
   - Consider computational resources
   - Test different methods on your specific dataset

2. **Language Support**
   - Verify method works for your language
   - SpaCy and YAKE offer best multilingual support

3. **Computational Complexity**
   - SpaCy and KeyBERT: More computationally intensive
   - TF-IDF and RAKE: Faster, less resource-heavy

## Troubleshooting

### Common Issues
- **Missing Libraries**: Install required packages
- **Poor Keyword Quality**: Adjust `max_ngram_size`
- **Language Mismatch**: Verify language-specific models

### Installation
```bash
# Core dependencies
pip install spacy scikit-learn

# Optional extraction methods
pip install rake-nltk
pip install yake
pip install keybert
```

## Best Practices

1. Experiment with different methods
2. Use appropriate stopwords
3. Tune `max_ngram_size`
4. Consider domain-specific requirements
5. Monitor keyword extraction performance

## Extending Keyword Extraction

- Create custom stopword lists
- Implement domain-specific filters
- Combine multiple extraction methods

## Logging and Monitoring

Enable debug logging to track keyword extraction:

```yaml
system:
  debug_level: 'debug'
```

## Security and Privacy

- Anonymize sensitive terms
- Implement access controls
- Be mindful of data processing regulations