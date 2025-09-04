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

import numpy as np
import structlog
from beartype.typing import Sequence
from typing_extensions import override

from superlinked.framework.common.space.embedding.model_based.embedding_input import (
    ModelEmbeddingInputT,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine import (
    EmbeddingEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.triton_engine_config import (
    TritonEngineConfig,
)

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    grpcclient = None
    InferenceServerException = Exception
    TRITON_AVAILABLE = False

logger = structlog.getLogger()


class TritonEngine(EmbeddingEngine[TritonEngineConfig]):
    """
    Triton Inference Server engine for embeddings using gRPC.
    
    This engine connects to a Triton Inference Server via gRPC to generate embeddings.
    It's designed to work with text embedding models deployed on Triton.
    """
    
    def __init__(self, model_name: str, model_cache_dir: Path | None, config: TritonEngineConfig) -> None:
        if not TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient is not available. Please install it using: "
                "pip install tritonclient[grpc]"
            )
        
        super().__init__(model_name, model_cache_dir, config)
        self._client = None
        self._model_config = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Triton gRPC client and validate model availability."""
        try:
            self._client = grpcclient.InferenceServerClient(
                url=self._config.triton_grpc_url,
                verbose=False
            )
            
            # Check if server is live
            if not self._client.is_server_live():
                raise ConnectionError(f"Triton server at {self._config.triton_grpc_url} is not live")
            
            # Check if model is ready
            if not self._client.is_model_ready(self._config.triton_model_name, self._config.triton_model_version):
                raise ValueError(
                    f"Model {self._config.triton_model_name} version {self._config.triton_model_version} "
                    f"is not ready on Triton server"
                )
            
            # Get model configuration
            self._model_config = self._client.get_model_config(
                self._config.triton_model_name, 
                self._config.triton_model_version
            )
            
            logger.info(
                "Triton client initialized successfully",
                grpc_url=self._config.triton_grpc_url,
                model_name=self._config.triton_model_name,
                model_version=self._config.triton_model_version
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize Triton client",
                error=str(e),
                grpc_url=self._config.triton_grpc_url,
                model_name=self._config.triton_model_name
            )
            raise

    @override
    async def embed(self, inputs: Sequence[ModelEmbeddingInputT], is_query_context: bool) -> list[list[float]]:
        """
        Generate embeddings using Triton Inference Server.
        
        Args:
            inputs: Sequence of text inputs to embed
            is_query_context: Whether this is a query context (for potential prompt optimization)
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if not inputs:
            return []
            
        # Convert inputs to list of strings
        text_inputs = [str(input_text) for input_text in inputs]
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._sync_embed, text_inputs)
        
        return embeddings.tolist()

    def _sync_embed(self, text_inputs: list[str]) -> np.ndarray:
        """
        Synchronous embedding generation using Triton client.
        
        Args:
            text_inputs: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Prepare input data
            input_data = np.array(text_inputs, dtype=object)
            
            # Create input object
            inputs = [
                grpcclient.InferInput("text", input_data.shape, "BYTES")
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            # Create output object
            outputs = [
                grpcclient.InferRequestedOutput("embeddings")
            ]
            
            # Perform inference with retries
            for attempt in range(self._config.triton_max_retries + 1):
                try:
                    response = self._client.infer(
                        model_name=self._config.triton_model_name,
                        model_version=self._config.triton_model_version,
                        inputs=inputs,
                        outputs=outputs,
                        timeout=int(self._config.triton_timeout_seconds)
                    )
                    
                    # Extract embeddings from response
                    embeddings = response.as_numpy("embeddings")
                    
                    logger.debug(
                        "Triton inference successful",
                        input_count=len(text_inputs),
                        embedding_shape=embeddings.shape,
                        attempt=attempt + 1
                    )
                    
                    return embeddings
                    
                except InferenceServerException as e:
                    if attempt == self._config.triton_max_retries:
                        logger.error(
                            "Triton inference failed after all retries",
                            error=str(e),
                            attempts=attempt + 1,
                            max_retries=self._config.triton_max_retries
                        )
                        raise
                    
                    logger.warning(
                        "Triton inference attempt failed, retrying",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=self._config.triton_max_retries
                    )
                    # Short delay before retry
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    
        except Exception as e:
            logger.error(
                "Error during Triton embedding generation",
                error=str(e),
                input_count=len(text_inputs)
            )
            raise

    def __del__(self) -> None:
        """Clean up the client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors
