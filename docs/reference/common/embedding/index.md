# Embedding Configuration

This section covers configuration and integration of different embedding backends with Superlinked.

## Available Backends

Superlinked supports multiple embedding backends:

- **SentenceTransformers** - Local model execution using sentence-transformers library
- **HuggingFace** - Direct integration with HuggingFace transformers
- **Modal** - Cloud-based inference via Modal platform
- **Triton** - High-performance inference server integration

## Configuration

Each backend has its own configuration options and settings. All backends support:

- Model caching
- Precision control (float16/float32)
- Timeout and retry settings
- Environment-based configuration

## Documentation

- [Triton Integration](triton-integration.md) - Complete guide for using Triton Inference Server

## Related

- [TextSimilaritySpace](../../dsl/space/text_similarity_space.md) - Main API for text embeddings
- [ImageSpace](../../dsl/space/image_space.md) - Image embedding configuration
- [Settings](../settings.md) - Global configuration management
