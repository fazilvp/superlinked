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
import time
from pathlib import Path

import numpy as np
import structlog
from beartype.typing import Sequence
from typing_extensions import override

from superlinked.framework.common.space.embedding.model_manager import (
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

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

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
        
        if config.triton_use_client_tokenizer and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not available. Please install it using: "
                "pip install transformers"
            )
        
        super().__init__(model_name, model_cache_dir, config)
        self._client = None
        self._model_config = None
        self._tokenizer = None
        self._initialize_client()
        self._initialize_tokenizer()

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

    def _initialize_tokenizer(self) -> None:
        """Initialize the tokenizer if client-side tokenization is enabled."""
        if not self._config.triton_use_client_tokenizer:
            return
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.triton_tokenizer_path
            )
            logger.info(
                "Tokenizer initialized successfully",
                tokenizer_path=self._config.triton_tokenizer_path
            )
        except Exception as e:
            logger.error(
                "Failed to initialize tokenizer",
                error=str(e),
                tokenizer_path=self._config.triton_tokenizer_path
            )
            raise

    @override
    async def embed(self, inputs: Sequence[ModelEmbeddingInputT], is_query_context: bool) -> list[list[float]]:
        """
        Generate embeddings using Triton Inference Server with batch processing.
        
        Args:
            inputs: Sequence of text inputs to embed
            is_query_context: Whether this is a query context (for potential prompt optimization)
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if not inputs:
            return []
        
        # Start total operation timing
        operation_start_time = time.perf_counter()
        
        # Convert inputs to list of strings
        text_inputs = [str(input_text) for input_text in inputs]
        
        # Split inputs into batches
        batches = [
            text_inputs[i : i + self._config.triton_batch_size] 
            for i in range(0, len(text_inputs), self._config.triton_batch_size)
        ]
        
        # Process batches concurrently
        loop = asyncio.get_event_loop()
        if self._config.triton_use_client_tokenizer:
            batch_results = await asyncio.gather(
                *[loop.run_in_executor(None, self._sync_embed_with_tokenizer, batch) for batch in batches]
            )
        else:
            batch_results = await asyncio.gather(
                *[loop.run_in_executor(None, self._sync_embed, batch) for batch in batches]
            )
        
        # Flatten results from all batches
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Calculate operation timing
        operation_end_time = time.perf_counter()
        total_operation_time = operation_end_time - operation_start_time
        
        logger.info(f"Triton inference: {total_operation_time * 1000:.2f}ms")
        
        return all_embeddings

    @override
    def is_query_prompt_supported(self) -> bool:
        """
        Triton engines typically don't support query-specific prompts unless specifically configured.
        
        Returns:
            False - indicating no query prompt support by default
        """
        return False

    def _sync_embed(self, text_inputs: list[str]) -> np.ndarray:
        """
        Synchronous embedding generation using Triton client (legacy mode for Python backend).
        
        Args:
            text_inputs: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Prepare input data
            input_data = np.array(text_inputs, dtype=object)
            
            # Create input object with correct batch dimension
            # For batched models with dims [-1], reshape to [batch_size, 1] for string inputs
            batch_size = len(text_inputs)
            input_data_reshaped = input_data.reshape(batch_size, 1)
            inputs = [
                grpcclient.InferInput("text", [batch_size, 1], "BYTES")
            ]
            inputs[0].set_data_from_numpy(input_data_reshaped)
            
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
                    time.sleep(0.1 * (attempt + 1))
                    
        except Exception as e:
            logger.error(
                "Error during Triton embedding generation",
                error=str(e),
                input_count=len(text_inputs)
            )
            raise

    def _sync_embed_with_tokenizer(self, text_inputs: list[str]) -> np.ndarray:
        """
        Synchronous embedding generation with client-side tokenization for ONNX models.
        
        Args:
            text_inputs: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Apply query format for embedding models
            formatted_texts = []
            for text in text_inputs:
                formatted_text = self._config.triton_instruction_template.format(text=text)
                formatted_texts.append(formatted_text)
            
            # Tokenize inputs
            tokenized = self._tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=self._config.triton_tokenizer_max_length,
                return_tensors="np"
            )
            
            # Prepare inputs for Triton
            inputs = [
                grpcclient.InferInput("input_ids", tokenized["input_ids"].shape, "INT64"),
                grpcclient.InferInput("attention_mask", tokenized["attention_mask"].shape, "INT64")
            ]
            inputs[0].set_data_from_numpy(tokenized["input_ids"].astype(np.int64))
            inputs[1].set_data_from_numpy(tokenized["attention_mask"].astype(np.int64))
            
            # Create output object
            outputs = [
                grpcclient.InferRequestedOutput("sentence_embedding_quantized")
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
                    
                    # Extract embeddings from response and convert from UINT8 to float32
                    embeddings_uint8 = response.as_numpy("sentence_embedding_quantized")
                    embeddings = embeddings_uint8.astype(np.float32) / 255.0
                    return embeddings
                    
                except InferenceServerException as e:
                    if attempt == self._config.triton_max_retries:
                        logger.error(
                            "Triton inference with tokenizer failed after all retries",
                            error=str(e),
                            attempts=attempt + 1,
                            max_retries=self._config.triton_max_retries
                        )
                        raise
                    
                    logger.warning(
                        "Triton inference with tokenizer attempt failed, retrying",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=self._config.triton_max_retries
                    )
                    # Short delay before retry
                    time.sleep(0.1 * (attempt + 1))
                    
        except Exception as e:
            logger.error(
                "Error during Triton embedding generation with tokenizer",
                error=str(e),
                input_count=len(text_inputs)
            )
            raise

    @classmethod
    @override
    def _get_clean_model_name(cls, model_name: str) -> str:
        """
        Return the clean model name for Triton models.
        
        Args:
            model_name: The original model name
            
        Returns:
            The model name as-is since Triton model names don't need preprocessing
        """
        return model_name

    def __del__(self) -> None:
        """Clean up the client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors
