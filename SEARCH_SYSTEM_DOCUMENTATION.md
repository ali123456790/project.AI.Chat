# Historical Letters AI Chatbot - Search System Documentation

## Overview
The Historical Letters AI Chatbot uses an intelligent conversational interface that automatically determines the best search method based on your natural language input. No more choosing between confusing search types - just ask naturally!

## System Configuration
- **Document Limit**: 200 historical letters (increased from 50)
- **AI Models Used**: 
  - spaCy for natural language processing
  - SentenceTransformers for semantic understanding
  - DistilBERT for question answering
  - Gensim for topic modeling

## How It Works

### 1. Query Classification System
When you type a question or statement, the system automatically analyzes it to determine the best search approach:

```python
class QueryClassifier:
    - Question patterns: "what", "who", "when", "where", "why", "how", "?"
    - Sender patterns: "from", "by", "written by", "authored by", "letter from"
    - Time patterns: Years (1800s-1900s), "in 1863", "during 1865"
    - Topic patterns: "about", "regarding", "military", "family", "business"
    - Keyword patterns: Short phrases (1-3 words)
```

### 2. Search Methods Available

#### A. Question Answering (QA) Search
**Triggers**: Questions with question words or question marks
- **Examples**: "What did soldiers write about battles?", "Who wrote about family?"
- **How it works**: 
  1. Uses semantic search to find relevant documents
  2. Applies DistilBERT QA model to extract specific answers
  3. Returns direct answers with confidence scores
- **Best for**: Direct factual questions about letter content

#### B. Semantic Search
**Triggers**: Complex sentences or when other patterns don't match
- **Examples**: "Letters about missing home", "Correspondence regarding military strategies"
- **How it works**:
  1. Converts your query into a 384-dimensional vector using SentenceTransformers
  2. Compares against pre-computed embeddings of all letters
  3. Finds documents with similar meaning, not just keywords
- **Best for**: Finding conceptually related content

#### C. Sender Search
**Triggers**: Patterns like "from John", "letters by Sarah", "written by"
- **Examples**: "Letters from John Smith", "Show me letters by Colonel Brown"
- **How it works**: Fuzzy matches against sender names in the database
- **Best for**: Finding all letters from specific people

#### D. Year/Time Search
**Triggers**: Year numbers (1800s-1900s) or time-related phrases
- **Examples**: "Letters from 1863", "What happened in 1865?"
- **How it works**: Filters documents by extracted year information
- **Best for**: Historical timeline exploration

#### E. Topic Search
**Triggers**: Words like "about", "regarding", military/family/business keywords
- **Examples**: "Military correspondence", "Personal family letters"
- **How it works**: Uses LDA topic modeling to find thematically similar documents
- **Best for**: Exploring specific themes across the collection

#### F. Keyword Search
**Triggers**: Short phrases (1-3 words) without common articles
- **Examples**: "Battle", "Money", "Home"
- **How it works**: Fast exact text matching across document content
- **Best for**: Finding specific mentions of terms

### 3. Advanced Search Features

#### Hybrid Search (Available via fallback)
Combines semantic and keyword search for comprehensive results:
- Uses TF-IDF for keyword relevance
- Merges with semantic similarity scores
- Re-ranks results for optimal relevance

#### Smart Query Processing
Uses spaCy NLP to extract:
- Named entities (people, places)
- Dates and time expressions
- Key phrases and relationships

## Technical Architecture

### Document Processing Pipeline
1. **XML Parsing**: Extracts structured data from TEI-encoded letters
2. **Text Preprocessing**: Cleans and normalizes content
3. **Embedding Generation**: Creates semantic vectors for each document
4. **Topic Modeling**: Discovers themes using LDA with 35 topics
5. **Indexing**: Creates searchable index with metadata

### Search Execution Flow
```
User Input → Query Classification → Search Method Selection → 
Document Retrieval → Result Ranking → Response Formatting
```

### Response Formatting
- **Questions**: Direct answers with confidence scores and source citations
- **Document searches**: Ranked list with relevant snippets
- **All results**: Include metadata (author, year, document source)

## Search Quality Features

### Confidence Scoring
Every search result includes confidence metrics:
- **QA**: Model confidence in answer accuracy (0-100%)
- **Semantic**: Cosine similarity scores
- **Topic**: Topic probability scores

### Result Ranking
Results are ranked by:
1. Relevance score (search-type specific)
2. Document quality indicators
3. Metadata completeness

### Error Handling
- Graceful fallbacks if primary search fails
- Alternative search suggestions
- Clear error messages with helpful hints

## Performance Optimizations

### Caching
- Pre-computed document embeddings
- Persistent topic models
- Indexed search structures

### Batch Processing
- Efficient similarity computations
- Vectorized operations where possible
- Memory-optimized data structures

## Usage Examples

### Natural Questions
- "What weapons were mentioned in letters?"
- "How did soldiers describe battles?"
- "Who wrote about missing their families?"

### Author-Based Searches
- "Letters from General Lee"
- "Show me correspondence by Sarah Johnson"

### Time-Based Searches
- "Letters from 1863"
- "What happened during 1865?"

### Topic Exploration
- "Military strategy discussions"
- "Personal family correspondence"
- "Business and trade letters"

### Keyword Searches
- "Cannon"
- "Hospital"
- "Richmond"

## Configuration Settings

Located in `chat.py` Config class:

```python
# Search Parameters
QA_MIN_SCORE_THRESHOLD = 0.1      # Minimum QA confidence
QA_MAX_ANSWER_LENGTH = 512         # Maximum answer length
SEMANTIC_TOP_N = 5                 # Number of semantic results
HYBRID_SEMANTIC_WEIGHT = 0.6       # Semantic vs keyword balance

# Performance Settings
MAX_FILES_FOR_TESTING = 200        # Document limit
BATCH_SIZE_EMBEDDING = 32          # Processing batch size
```

## Future Enhancements

### Planned Features
- **Multi-document QA**: Answers spanning multiple letters
- **Timeline visualization**: Visual representation of search results over time
- **Advanced filtering**: By document type, location, sentiment
- **Export capabilities**: Save search results and conversations

### Potential Improvements
- **Context-aware follow-ups**: Remember conversation history
- **Fuzzy matching**: Better handling of OCR errors
- **Relationship extraction**: Find connections between people and events
- **Sentiment analysis**: Understand emotional tone of letters

## Troubleshooting

### Common Issues
1. **No results found**: Try different keywords or rephrase as a question
2. **QA not working**: Ensure question is specific and factual
3. **Slow responses**: Large semantic searches may take time
4. **Unclear results**: Add more context to your query

### Getting Better Results
- Use specific names when searching for people
- Include context words for better semantic matching
- Try both question and keyword formats
- Use year ranges for temporal searches

This search system represents a significant advancement in historical document exploration, making Civil War era letters accessible through natural conversation rather than complex search interfaces. 