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
from typing import Any, Dict

from typing_extensions import override

from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine_config import (
    EmbeddingEngineConfig,
)


@dataclass(frozen=True, kw_only=True)
class FastEmbedEngineConfig(EmbeddingEngineConfig):
    """
    Configuration for FastEmbed embedding engine.
    
    Args:
        custom_model_config (Dict[str, Any], optional): Configuration for custom models.
            Contains model_file, sources, and other FastEmbed model parameters.
            Defaults to None.
        max_length (int, optional): Maximum sequence length for embeddings.
            Defaults to 512.
        cache_dir (str, optional): Directory to cache downloaded models.
            If None, uses FastEmbed's default cache. Defaults to None.
        threads (int, optional): Number of threads for ONNX runtime.
            Defaults to None (auto-detect).
        providers (list, optional): ONNX execution providers.
            Defaults to None (auto-select).
    """

    custom_model_config: Dict[str, Any] | None = None
    max_length: int = 512
    cache_dir: str | None = None
    threads: int | None = None
    providers: list[str] | None = None

    @override
    def __str__(self) -> str:
        parts = [
            self.precision.name,
            f"max_length={self.max_length}",
        ]
        if self.custom_model_config:
            parts.append("custom_model=True")
        if self.cache_dir:
            parts.append(f"cache_dir={self.cache_dir}")
        if self.threads:
            parts.append(f"threads={self.threads}")
        return ":".join(parts)