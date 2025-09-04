# Triton Integration for Superlinked Embeddings

This document describes how to use Triton Inference Server with Superlinked for high-performance embedding generation.

## Overview

The Triton integration allows you to use Triton Inference Server as the backend for generating embeddings in Superlinked, providing:

- **High Performance**: Optimized inference with batching and GPU acceleration
- **Scalability**: Horizontal scaling with multiple Triton instances
- **Production Ready**: Battle-tested inference server for production workloads
- **Flexibility**: Support for various model formats and deployment options

## Prerequisites

### 1. Install Triton Client

```bash
pip install tritonclient[grpc]
```

### 2. Running Triton Server

You need a Triton server with an embedding model deployed. For example, using the provided Docker setup:

```bash
# From the qwen-deploy directory
docker-compose up -d
```

This starts Triton on:
- gRPC: `localhost:6565` (recommended for low latency)
- HTTP: `localhost:8000`

### 3. Verify Triton Setup

Check that your model is loaded:

```bash
curl http://localhost:8000/v2/models/qwen3-embedding-06B/ready
```

## Usage

### Basic Usage

```python
from superlinked import (
    TextSimilaritySpace, 
    TextModelHandler, 
    TritonEngineConfig,
    Schema, 
    String
)

# Define your schema
@schema
class Document:
    id: str
    content: String

# Configure Triton connection - uses Settings defaults
triton_config = TritonEngineConfig()

# Create TextSimilaritySpace with Triton backend
text_space = TextSimilaritySpace(
    text=Document.content,
    model="qwen3-embedding-06B",
    model_handler=TextModelHandler.TRITON,
    embedding_engine_config=triton_config,
    cache_size=1000
)
```

### Configuration Sources

Triton settings are read from multiple sources in order of priority:

1. **Direct parameter overrides** (highest priority)
2. **Environment variables** (e.g., `TRITON_GRPC_URL`)
3. **config.yaml file** (framework section)
4. **Built-in defaults** (lowest priority)

### Configuration via config.yaml

Create a `config.yaml` file in your project root:

```yaml
framework:
  # Triton settings
  TRITON_GRPC_URL: "localhost:6565"
  TRITON_MODEL_NAME: "qwen3-embedding-06B"
  TRITON_MODEL_VERSION: "1"
  TRITON_TIMEOUT_SECONDS: 60.0
  TRITON_MAX_RETRIES: 3
```

### Configuration via Environment Variables

```bash
export TRITON_GRPC_URL="localhost:6565"
export TRITON_MODEL_NAME="qwen3-embedding-06B"
export TRITON_MODEL_VERSION="1"
export TRITON_TIMEOUT_SECONDS="60.0"
export TRITON_MAX_RETRIES="3"
```

### Configuration Options

#### TritonEngineConfig Parameters

| Parameter | Type | Default (Settings) | Description |
|-----------|------|---------|-------------|
| `grpc_url` | str\|None | `Settings.TRITON_GRPC_URL` | Triton gRPC endpoint |
| `model_name` | str\|None | `Settings.TRITON_MODEL_NAME` | Model name on Triton |
| `model_version` | str\|None | `Settings.TRITON_MODEL_VERSION` | Model version |
| `timeout_seconds` | float\|None | `Settings.TRITON_TIMEOUT_SECONDS` | Request timeout |
| `max_retries` | int\|None | `Settings.TRITON_MAX_RETRIES` | Maximum retry attempts |
| `precision` | Precision | `FLOAT16` | Embedding precision |

#### Example Configurations

**Use all defaults (from Settings):**
```python
config = TritonEngineConfig()
```

**Override specific values:**
```python
config = TritonEngineConfig(
    grpc_url="triton.your-domain.com:443",  # Override URL
    timeout_seconds=120.0  # Override timeout
    # Other values come from Settings
)
```

**Production override:**
```python
prod_config = TritonEngineConfig(
    grpc_url="triton-prod.company.com:443",
    model_name="production-embedding-model",
    model_version="2",
    timeout_seconds=60.0,
    max_retries=5
)
```

### Complete Example

```python
import asyncio
from superlinked import *

@schema
class Article:
    id: str
    title: String
    content: String

# Configure Triton
triton_config = TritonEngineConfig(
    grpc_url="localhost:6565",
    model_name="qwen3-embedding-06B"
)

# Create embedding space
article_space = TextSimilaritySpace(
    text=[Article.title, Article.content],
    model="qwen3-embedding-06B",
    model_handler=TextModelHandler.TRITON,
    embedding_engine_config=triton_config
)

# Create index
index = Index(article_space)

# Set up executor
executor = InMemoryExecutor(
    sources=[InMemorySource(Article)],
    indices=[index]
)

# Add data
articles = [
    Article(id="1", title="AI Revolution", content="Artificial intelligence is transforming..."),
    Article(id="2", title="ML Basics", content="Machine learning fundamentals include..."),
]

app = executor.run()
for article in articles:
    app.source.put([article])

# Query
result = app.query(
    Query(article_space).similar("machine learning", weight=1.0),
    limit=5
)
```

## Performance Considerations

### 1. Batch Size
Triton automatically batches requests for optimal performance. The embedding engine handles this transparently.

### 2. Connection Pooling
Each `TritonEngine` instance maintains its own gRPC connection. For high-concurrency scenarios, consider:
- Using connection pooling at the application level
- Deploying multiple Triton replicas with load balancing

### 3. Caching
Enable embedding caching in TextSimilaritySpace:
```python
text_space = TextSimilaritySpace(
    # ... other parameters
    cache_size=10000  # Cache 10k embeddings in memory
)
```

### 4. Error Handling
The engine includes automatic retry logic with exponential backoff:
- Configurable via `max_retries`
- Automatic failover for transient errors
- Detailed logging for debugging

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Failed to initialize Triton client
   ```
   - Check if Triton server is running
   - Verify the gRPC URL and port
   - Ensure firewall allows connections

2. **Model Not Ready**
   ```
   ValueError: Model qwen3-embedding-06B version 1 is not ready
   ```
   - Check model deployment status
   - Verify model name and version
   - Check Triton server logs

3. **Import Error**
   ```
   ImportError: tritonclient is not available
   ```
   - Install tritonclient: `pip install tritonclient[grpc]`

### Debugging

Enable detailed logging:
```python
import structlog
structlog.configure(level="DEBUG")
```

Check Triton server status:
```bash
# Health check
curl http://localhost:8000/v2/health/ready

# Model status
curl http://localhost:8000/v2/models/qwen3-embedding-06B

# Server metadata
curl http://localhost:8000/v2
```

## Migration Guide

### From SentenceTransformers

**Before:**
```python
text_space = TextSimilaritySpace(
    text=Document.content,
    model="sentence-transformers/all-mpnet-base-v2",
    model_handler=TextModelHandler.SENTENCE_TRANSFORMERS
)
```

**After:**
```python
triton_config = TritonEngineConfig(
    grpc_url="localhost:6565",
    model_name="all-mpnet-base-v2"  # Deployed on Triton
)

text_space = TextSimilaritySpace(
    text=Document.content,
    model="all-mpnet-base-v2",
    model_handler=TextModelHandler.TRITON,
    embedding_engine_config=triton_config
)
```

### Migration Checklist

- [ ] Deploy your model to Triton server
- [ ] Install `tritonclient[grpc]`
- [ ] Update model handler to `TextModelHandler.TRITON`
- [ ] Add `TritonEngineConfig` with your server details
- [ ] Test with small dataset first
- [ ] Monitor performance and adjust configuration

## Best Practices

1. **Use gRPC for Performance**: gRPC provides better performance than HTTP for high-frequency requests

2. **Configure Timeouts**: Set appropriate timeouts based on your model and expected load

3. **Monitor Resource Usage**: Watch Triton server CPU/GPU/memory usage

4. **Implement Circuit Breakers**: For production, implement circuit breakers around Triton calls

5. **Version Your Models**: Use explicit model versions for reproducible results

6. **Load Testing**: Test your setup under expected load before production deployment

## Support

For issues specific to:
- **Triton Server**: Check [NVIDIA Triton documentation](https://github.com/triton-inference-server/server)
- **Superlinked Integration**: Create an issue in the Superlinked repository
- **Model Deployment**: Refer to your model's deployment guide
