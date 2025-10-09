# Copyright 2024 Superlinked, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from pathlib import Path

import structlog
from beartype.typing import Sequence, cast
from typing_extensions import override

from superlinked.framework.common.space.embedding.model_based.embedding_input import (
    ModelEmbeddingInput,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine import (
    EmbeddingEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.fastembed_engine_config import (
    FastEmbedEngineConfig,
)

logger = structlog.getLogger()


class FastEmbedEngine(EmbeddingEngine[FastEmbedEngineConfig]):
    def __init__(self, model_name: str, model_cache_dir: Path | None, config: FastEmbedEngineConfig) -> None:
        super().__init__(model_name, model_cache_dir, config)
        self._model = None
        self._initialized = False

    @override
    async def embed(self, inputs: Sequence[ModelEmbeddingInput], is_query_context: bool) -> list[list[float]]:
        """
        Embed the input texts using FastEmbed.
        
        Args:
            inputs: Sequence of text inputs to embed
            is_query_context: Whether this is a query context (unused in FastEmbed)
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self._initialize_model()

        # Convert ModelEmbeddingInput to strings
        text_inputs = [cast(str, input_) for input_ in inputs if isinstance(input_, str)]
        
        if not text_inputs:
            return []

        def sync_embed() -> list[list[float]]:
            # Get embeddings from FastEmbed model
            embeddings = list(self._model.embed(text_inputs))
            return [embedding.tolist() for embedding in embeddings]

        return await asyncio.to_thread(sync_embed)

    @override
    def is_query_prompt_supported(self) -> bool:
        """FastEmbed doesn't use query prompts."""
        return False

    async def _initialize_model(self) -> None:
        """Initialize the FastEmbed model with custom configuration if provided."""
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource
        except ImportError as e:
            raise ImportError(
                "FastEmbed is not installed. Please install it with: pip install fastembed"
            ) from e

        def sync_init() -> None:
            # Add custom model if configured
            if self._config.custom_model_config:
                custom_config = self._config.custom_model_config
                model_name = custom_config.get("model", self._model_name)
                model_file = custom_config.get("model_file")
                sources = custom_config.get("sources")
                
                if model_file and sources:
                    # Add custom model to FastEmbed registry
                    TextEmbedding.add_custom_model(
                        model=model_name,
                        model_file=model_file,
                        sources=ModelSource(**sources) if isinstance(sources, dict) else sources,
                    )
                    logger.info(f"Added custom FastEmbed model: {model_name}")

            # Initialize the model with configuration
            init_kwargs = {
                "model_name": self._model_name,
                "max_length": self._config.max_length,
            }
            
            if self._config.cache_dir:
                init_kwargs["cache_dir"] = self._config.cache_dir
            elif self._model_cache_dir:
                init_kwargs["cache_dir"] = str(self._model_cache_dir)
                
            if self._config.threads:
                init_kwargs["threads"] = self._config.threads
                
            if self._config.providers:
                init_kwargs["providers"] = self._config.providers

            self._model = TextEmbedding(**init_kwargs)
            logger.info(f"Initialized FastEmbed model: {self._model_name}")

        await asyncio.to_thread(sync_init)
        self._initialized = True

    @classmethod
    @override
    def _get_clean_model_name(cls, model_name: str) -> str:
        """Return the clean model name for key generation."""
        return f"fastembed/{model_name}"