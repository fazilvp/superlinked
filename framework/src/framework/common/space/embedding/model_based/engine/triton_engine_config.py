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

from dataclasses import dataclass

from typing_extensions import override

from superlinked.framework.common.precision import Precision
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine_config import (
    EmbeddingEngineConfig,
)


@dataclass(frozen=True)
class TritonEngineConfig(EmbeddingEngineConfig):
    """
    Configuration for Triton Inference Server engine.
    
    Args:
        grpc_url (str): The gRPC URL of the Triton server (e.g., "localhost:8001").
        model_name (str): The model name deployed on Triton server.
        model_version (str): The model version (defaults to "1").
        timeout_seconds (float): Request timeout in seconds (defaults to 60.0).
        max_retries (int): Maximum number of retries (defaults to 3).
        batch_size (int): Maximum batch size for processing inputs (defaults to 32).
        precision (Precision, optional): The desired precision for embeddings. 
                                        Defaults to Precision.FLOAT16.
    """
    
    grpc_url: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    batch_size: int | None = None
    
    def __post_init__(self) -> None:
        # Validate configuration values
        if not self.grpc_url:
            raise ValueError("grpc_url cannot be empty")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def triton_grpc_url(self) -> str:
        """Get the gRPC URL."""
        return self.grpc_url or "localhost:8001"

    @property
    def triton_model_name(self) -> str:
        """Get the model name."""
        return self.model_name or "default_model"

    @property
    def triton_model_version(self) -> str:
        """Get the model version."""
        return self.model_version or "1"

    @property
    def triton_timeout_seconds(self) -> float:
        """Get the timeout in seconds."""
        return self.timeout_seconds or 60.0

    @property
    def triton_max_retries(self) -> int:
        """Get the maximum number of retries."""
        return self.max_retries or 3

    @property
    def triton_batch_size(self) -> int:
        """Get the batch size."""
        return self.batch_size or 32

    @override
    def __str__(self) -> str:
        return f"{self.precision.name}_{self.grpc_url}_{self.model_name}_{self.model_version}_{self.batch_size}"
