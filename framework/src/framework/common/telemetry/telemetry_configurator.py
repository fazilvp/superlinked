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

from superlinked.framework.common.settings import Settings
from superlinked.framework.common.telemetry.telemetry_registry import (
    MetricType,
    TelemetryRegistry,
)


class SuperlinkedTelemetryConfigurator:
    @staticmethod
    def configure_default_metrics() -> None:
        settings = Settings()
        telemetry = TelemetryRegistry()

        labels = {
            "app_id": settings.APP_ID,
        }
        telemetry.add_labels(labels)

        telemetry.create_metric(MetricType.COUNTER, "embeddings_total", "Total number of embeddings calculated", "1")
