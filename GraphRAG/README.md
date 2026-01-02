# 🚀 Agentic RAG Platform with LangGraph

A production-grade, stateful Retrieval-Augmented Generation system built with LangGraph that orchestrates multi-step AI workflows with conditional branching, validation loops, and intelligent document processing.

## 🧠 Core Architecture

Instead of traditional linear chains, this system implements a sophisticated graph-based workflow:

```
User Query
    ↓
Query Analyzer (Intent Detection & Query Rewriting)
    ↓
Retriever (Vector/Hybrid/Keyword Search)
    ↓
[Conditional Edge: Enough Evidence?] ──→ Query Rewriter (if needed)
    ↓
Re-Ranker (Cross-encoder & LLM-based scoring)
    ↓
Answer Generator (Context-aware, grounded responses)
    ↓
Answer Validator (Hallucination detection & quality checks)
    ↓
Final Response
```

## ✨ Key Features

### 🎯 **LangGraph Workflow**
- **Stateful Processing**: Maintains rich state across workflow steps
- **Conditional Branching**: Intelligent decision-making at each step
- **Retry Logic**: Automatic retries with query rewriting when retrieval quality is poor
- **Validation Loops**: Answer regeneration when quality checks fail

### 📚 **Advanced Document Ingestion**
- **Adaptive Chunking**: Semantic, fixed-size, and hybrid strategies
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, HTML
- **Quality Filtering**: Automatic removal of low-quality content
- **Metadata Enrichment**: Rich document metadata for better retrieval

### 🔍 **Intelligent Retrieval**
- **Hybrid Search**: Vector + BM25 keyword search
- **Query Analysis**: Intent detection and query expansion
- **Dynamic Top-K**: Adaptive document selection
- **Metadata Filtering**: Date, source, and type-based filtering

### 🛡️ **Quality Assurance**
- **Answer Validation**: Groundedness and hallucination detection
- **Confidence Scoring**: Multi-factor confidence calculation
- **Citation Management**: Automatic source citation and highlighting
- **Guardrail Agents**: Content safety and accuracy checks

### ⚡ **Production Optimizations**
- **Redis Caching**: Multi-layer caching for queries, embeddings, and retrieval
- **Streaming Responses**: Real-time workflow progress streaming
- **Prometheus Metrics**: Comprehensive monitoring and alerting
- **Async Processing**: Non-blocking document ingestion and querying

## 🏗️ System Components

### 1. **Document Ingestion Pipeline**
```python
# Adaptive chunking with multiple strategies
pipeline = DocumentIngestionPipeline()
result = await pipeline.ingest_document("document.pdf", strategy="adaptive")
```

### 2. **LangGraph Workflow**
```python
# Multi-step RAG workflow with state management
workflow = RAGWorkflow()
result = await workflow.run("What are the benefits of LangGraph?")
```

### 3. **FastAPI Backend**
```python
# Production-ready API endpoints
# POST /api/v1/query - Process queries
# POST /api/v1/upload - Ingest documents
# GET /api/v1/sources - List document sources
# POST /api/v1/feedback - Submit user feedback
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Redis (for caching)
- OpenAI API key

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/nengihart-X/GraphRAG-.git
cd GraphRAG-/GraphRAG
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Start Redis**
```bash
# On Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# On macOS
brew install redis
brew services start redis

# On Windows
# Download and install Redis from https://redis.io/download
```

4. **Run the Application**
```bash
python -m src.api.main
```

The API will be available at `http://localhost:8000`

### 📖 API Usage

#### Query Processing
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of LangGraph?",
    "session_id": "user123"
  }'
```

#### Document Upload
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=adaptive"
```

#### Streaming Query
```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain RAG architecture",
    "stream": true
  }'
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Vector Database
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./data/vectors

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNK_SIZE=1024

# Retrieval Settings
DEFAULT_TOP_K=10
RETRIEVAL_THRESHOLD=0.7
MAX_RETRIEVAL_ATTEMPTS=3
```

## 📊 Monitoring & Metrics

### Prometheus Metrics
The system exposes comprehensive metrics at `/api/v1/metrics`:

- **Query Metrics**: Duration, confidence, intent distribution
- **Retrieval Metrics**: Document counts, scores, strategy usage
- **Validation Metrics**: Pass rates, hallucination scores
- **System Metrics**: Cache performance, active sessions

### Health Checks
```bash
curl http://localhost:8000/api/v1/health
```

### System Dashboard
Access real-time metrics and system health at:
- API Documentation: `http://localhost:8000/docs`
- Metrics Endpoint: `http://localhost:8000/api/v1/metrics`

For advanced monitoring, you can optionally set up Prometheus:
```bash
# Install Prometheus (optional)
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvfz prometheus-2.40.0.linux-amd64.tar.gz
cd prometheus-2.40.0.linux-amd64
./prometheus --config.file=prometheus.yml
```

## 🧪 Advanced Features

### Multi-Agent RAG
The system supports multiple specialized agents:
- **Research Agent**: Deep document analysis
- **Answer Agent**: Response generation
- **Critic Agent**: Quality validation
- **Tool Agent**: External API integration

### Conversation Memory
Maintains conversation context across sessions:
```python
# Session-based conversation memory
result = await workflow.run(
    "Follow up on that previous point",
    session_id="conversation_123"
)
```

### Tool Integration
Extensible tool system for external data sources:
- Web search fallback
- Database queries
- API integrations
- Custom tools

## 🏭 Production Deployment

### Manual Deployment
1. Install system dependencies:
```bash
# Install Python 3.9+
sudo apt install python3.9 python3.9-venv python3-pip

# Install Redis
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

2. Setup production environment:
```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with production settings
```

3. Run with process manager:
```bash
# Using gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app

# Or using systemd (create service file)
sudo nano /etc/systemd/system/rag-platform.service
```

### Production Configuration
```bash
# Environment variables for production
export DEBUG=false
export LOG_LEVEL=INFO
export API_HOST=0.0.0.0
export API_PORT=8000
```

### Scaling Considerations
- **Horizontal Scaling**: Multiple instances behind nginx load balancer
- **Vector Database**: Scale ChromaDB or migrate to Pinecone/Weaviate
- **Cache Layer**: Redis Cluster for distributed caching
- **Monitoring**: Prometheus + Grafana for observability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the excellent LangGraph framework
- **OpenAI** for powerful language models
- **ChromaDB** for efficient vector storage
- **FastAPI** for modern API development

---

## 📞 Support

For questions and support:
- 📧 Email: nengihaart2006@gmail.com

Built with ❤️ using LangGraph

