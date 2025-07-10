# 📜 AI-HIS-Chatbot - Historical Letters AI Analysis System

> **Advanced AI-powered chatbot for analyzing Civil War era correspondence using state-of-the-art BGE models and semantic search**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Features

### 🤖 **State-of-the-Art AI Models**
- **BAAI/bge-large-en-v1.5** - Superior semantic document retrieval
- **BAAI/bge-reranker-large** - Precision context ranking and filtering
- **deepset/roberta-base-squad2** - Advanced question answering
- **Google Gemini** - Natural language conversation enhancement

### 📊 **Advanced Search Capabilities**
- **Semantic Search** - Find documents by meaning, not just keywords
- **Question Answering** - Direct answers extracted from historical letters
- **Cross-Encoder Re-ranking** - Precision relevance scoring
- **Faceted Filtering** - Search by date, sender, location, topic
- **Topic Modeling** - Discover themes across 50+ automatically detected topics

### 🗺️ **Geographic Visualization**
- **Interactive Maps** - Visualize letter locations with clustering
- **Coordinate Extraction** - Parse `<geo>` tags from TEI XML
- **Geocoding Fallback** - Automatic location lookup with geopy
- **Heat Maps** - Density visualization of correspondence patterns

### 📄 **Comprehensive Data Processing**
- **5,314+ Civil War Letters** - Complete TEI XML dataset
- **Enhanced XML Parsing** - Extract metadata, coordinates, taxonomy
- **OCR Quality Filtering** - Remove scanning artifacts
- **Historical Stop-word Processing** - Period-appropriate text analysis

## 🏃‍♂️ Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements_working.txt
```

### Installation
```bash
git clone https://github.com/ali123456790/AI-HIS-Chatbot.git
cd AI-HIS-Chatbot
pip install -r requirements_working.txt
```

### Run the Application
```bash
streamlit run app_gui.py
```

Open your browser to `http://localhost:8501`

## 🔧 Configuration

### API Keys
Set your Google Gemini API key in `chat.py`:
```python
GEMINI_API_KEY = "your-api-key-here"
```

### Model Settings
Adjust AI models in `chat.py`:
- `SENTENCE_MODEL_NAME` - BGE embedding model
- `CROSS_ENCODER_MODEL_NAME` - BGE reranker model  
- `QA_MODEL_NAME` - RoBERTa QA model

## 📖 Usage Examples

### Semantic Search
```
"How did soldiers describe battle conditions?"
"Letters about family concerns during wartime"
"Correspondence discussing military strategy"
```

### Geographic Queries
```
"Letters from Virginia battlefields"
"Correspondence from Union camps"
"Mail sent from Southern states"
```

### Temporal Analysis
```
"How did morale change over time?"
"Letters from 1863 Gettysburg campaign"
"Correspondence during Sherman's March"
```

## 🏗️ Architecture

```
User Query → BGE Retrieval → BGE Reranking → RoBERTa QA → Gemini Enhancement
     ↓
TEI XML Documents (5,314+) → Embeddings → Vector Search → Answer Extraction
```

### Processing Pipeline
1. **XML Parsing** - Extract text, metadata, coordinates from TEI
2. **Text Preprocessing** - OCR cleanup, historical stop-words
3. **Embedding Generation** - BGE-large semantic vectors
4. **Index Creation** - Optimized vector database
5. **Query Processing** - Multi-stage retrieval and ranking

## 📁 Project Structure

```
AI-HIS-Chatbot/
├── app_gui.py              # Streamlit web interface
├── chat.py                 # Core AI processing logic
├── requirements_working.txt # Python dependencies
├── run_app.sh             # Launch script
├── renamed_highlighted_cwrgm_xml/ # TEI XML letter corpus
│   ├── mdah_*.xml         # Individual letter files
│   └── ...
└── README.md              # This file
```

## 🔍 Search System Features

### Multi-Modal Search
- **Keyword Search** - Traditional text matching
- **Semantic Search** - BGE embedding similarity
- **Question Answering** - Direct answer extraction
- **Topic Search** - Thematic document clusters
- **Geographic Search** - Location-based filtering

### Advanced Filtering
- **Date Ranges** - Year-based temporal filtering
- **Person Names** - Sender/recipient search
- **Geographic Regions** - Location-based results
- **Document Types** - Letter classification filtering
- **Topic Categories** - Thematic content grouping

## 🗺️ Map Visualization

### Interactive Features
- **Marker Clustering** - Grouped geographic points
- **Color-coded Density** - Visual correspondence volume
- **Multi-layer Maps** - Terrain, satellite, street views
- **Heat Map Overlay** - Geographic distribution patterns
- **Popup Details** - Letter metadata on click

### Coordinate Sources
1. **Direct XML `<geo>` tags** - Precise coordinates
2. **Historical location lookup** - Curated database
3. **Live geocoding** - Automatic location resolution

## 🤖 AI Model Details

### BGE Retrieval Stack
- **Embeddings**: BAAI/bge-large-en-v1.5 (1.3GB)
- **Reranking**: BAAI/bge-reranker-large
- **Performance**: SOTA retrieval on MS-MARCO benchmark

### Question Answering
- **Model**: deepset/roberta-base-squad2
- **Capabilities**: Context-aware answer extraction
- **Confidence Scoring**: Quality-based filtering

### Natural Language Enhancement
- **Integration**: Google Gemini 1.5 Flash
- **Purpose**: Conversational response generation
- **Features**: Context-aware, historically informed

## 📊 Performance Metrics

- **Document Corpus**: 5,314+ Civil War letters
- **Processing Speed**: ~1-2 hours for full indexing
- **Search Latency**: <2 seconds for complex queries
- **Geographic Coverage**: 1,000+ unique locations
- **Topic Detection**: 50+ automatically discovered themes

## 🔧 Technical Requirements

### Hardware
- **RAM**: 8GB+ recommended for full dataset
- **Storage**: 10GB+ for models and data
- **CPU**: Multi-core recommended for embedding generation

### Software Dependencies
```
streamlit>=1.28.0
sentence-transformers>=2.2.0
transformers>=4.21.0
torch>=1.12.0
google-generativeai>=0.3.0
geopy>=2.3.0
folium>=0.14.0
spacy>=3.4.0
gensim>=4.2.0
scikit-learn>=1.1.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License


## 🙏 Acknowledgments

- **Mississippi Department of Archives & History** - TEI XML letter corpus
- **Hugging Face** - AI model hosting and transformers library
- **BAAI** - BGE model development
- **Google** - Gemini AI integration
- **Streamlit** - Web application framework

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation in `SEARCH_SYSTEM_DOCUMENTATION.md`
- Review the code comments for implementation details

---

**Built with ❤️ for historical research and AI innovation** 
