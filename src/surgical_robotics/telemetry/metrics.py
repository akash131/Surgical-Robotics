"""Metrics collection and export for surgical robotics."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable
import time
import threading
from collections import deque
import json


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = auto()  # Monotonically increasing
    GAUGE = auto()  # Current value
    HISTOGRAM = auto()  # Distribution of values
    SUMMARY = auto()  # Statistical summary


@dataclass
class Metric:
    """Single metric definition."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    # Current value(s)
    value: float = 0.0
    values: List[float] = field(default_factory=list)

    # Statistics (for histogram/summary)
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')

    # Histogram buckets
    buckets: Dict[float, int] = field(default_factory=dict)

    # Timestamp
    last_update: float = 0.0

    def record(self, value: float) -> None:
        """Record a value."""
        self.last_update = time.time()

        if self.metric_type == MetricType.COUNTER:
            self.value += value
        elif self.metric_type == MetricType.GAUGE:
            self.value = value
        elif self.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            self.values.append(value)
            self.count += 1
            self.sum += value
            self.min = min(self.min, value)
            self.max = max(self.max, value)

            # Update buckets
            for bucket in sorted(self.buckets.keys()):
                if value <= bucket:
                    self.buckets[bucket] += 1

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter."""
        if self.metric_type == MetricType.COUNTER:
            self.record(amount)

    def set(self, value: float) -> None:
        """Set gauge value."""
        if self.metric_type == MetricType.GAUGE:
            self.record(value)

    def observe(self, value: float) -> None:
        """Observe value for histogram/summary."""
        if self.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            self.record(value)

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def reset(self) -> None:
        """Reset metric."""
        if self.metric_type == MetricType.COUNTER:
            return  # Counters don't reset
        self.value = 0.0
        self.values.clear()
        self.count = 0
        self.sum = 0.0
        self.min = float('inf')
        self.max = float('-inf')


class MetricsCollector:
    """
    Collects and manages metrics for surgical robotics.

    Features:
    - Multiple metric types (counter, gauge, histogram)
    - Labels for dimensionality
    - Time-series storage
    - Prometheus-compatible export
    """

    def __init__(self, prefix: str = "surgical_robot") -> None:
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

        # Time series data (limited buffer)
        self._time_series: Dict[str, deque] = {}
        self._series_max_length = 10000

        # Collection callbacks
        self._collectors: List[Callable[[], Dict[str, float]]] = []

    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Register a counter metric."""
        return self._register(
            name,
            MetricType.COUNTER,
            description,
            labels or {}
        )

    def register_gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Register a gauge metric."""
        metric = self._register(
            name,
            MetricType.GAUGE,
            description,
            labels or {}
        )
        metric.unit = unit
        return metric

    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Register a histogram metric."""
        metric = self._register(
            name,
            MetricType.HISTOGRAM,
            description,
            labels or {}
        )
        # Default buckets
        bucket_values = buckets or [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        metric.buckets = {b: 0 for b in bucket_values}
        return metric

    def _register(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Dict[str, str]
    ) -> Metric:
        """Internal registration."""
        full_name = f"{self.prefix}_{name}"

        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]

            metric = Metric(
                name=full_name,
                metric_type=metric_type,
                description=description,
                labels=labels
            )
            self._metrics[full_name] = metric
            self._time_series[full_name] = deque(maxlen=self._series_max_length)

        return metric

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        full_name = f"{self.prefix}_{name}"
        return self._metrics.get(full_name)

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment counter."""
        metric = self.get_metric(name)
        if metric:
            metric.increment(amount)
            self._record_series(metric)

    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value."""
        metric = self.get_metric(name)
        if metric:
            metric.set(value)
            self._record_series(metric)

    def observe(self, name: str, value: float) -> None:
        """Observe histogram/summary value."""
        metric = self.get_metric(name)
        if metric:
            metric.observe(value)
            self._record_series(metric)

    def _record_series(self, metric: Metric) -> None:
        """Record to time series."""
        with self._lock:
            if metric.name in self._time_series:
                self._time_series[metric.name].append(
                    (time.time(), metric.value)
                )

    def add_collector(
        self,
        collector: Callable[[], Dict[str, float]]
    ) -> None:
        """Add custom metric collector."""
        self._collectors.append(collector)

    def collect(self) -> Dict[str, Any]:
        """Collect all metrics."""
        result = {}

        with self._lock:
            for name, metric in self._metrics.items():
                result[name] = {
                    "type": metric.metric_type.name,
                    "value": metric.value,
                    "labels": metric.labels,
                    "description": metric.description,
                    "unit": metric.unit,
                    "last_update": metric.last_update
                }

                if metric.metric_type == MetricType.HISTOGRAM:
                    result[name].update({
                        "count": metric.count,
                        "sum": metric.sum,
                        "min": metric.min if metric.min != float('inf') else 0,
                        "max": metric.max if metric.max != float('-inf') else 0,
                        "buckets": metric.buckets
                    })

        # Run custom collectors
        for collector in self._collectors:
            try:
                custom_metrics = collector()
                for name, value in custom_metrics.items():
                    result[f"{self.prefix}_{name}"] = {
                        "type": "GAUGE",
                        "value": value
                    }
            except Exception:
                pass

        return result

    def get_time_series(
        self,
        name: str,
        since: Optional[float] = None
    ) -> List[tuple]:
        """Get time series data for metric."""
        full_name = f"{self.prefix}_{name}"

        with self._lock:
            if full_name not in self._time_series:
                return []

            series = list(self._time_series[full_name])

            if since:
                series = [(t, v) for t, v in series if t >= since]

            return series


class MetricsExporter:
    """
    Exports metrics to various formats and backends.

    Supports:
    - Prometheus format
    - JSON
    - StatsD
    """

    def __init__(self, collector: MetricsCollector) -> None:
        self.collector = collector
        self._export_interval = 10.0  # seconds
        self._running = False
        self._export_thread: Optional[threading.Thread] = None

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.collector.collect()

        for name, data in metrics.items():
            # Help line
            if data.get("description"):
                lines.append(f"# HELP {name} {data['description']}")

            # Type line
            prom_type = {
                "COUNTER": "counter",
                "GAUGE": "gauge",
                "HISTOGRAM": "histogram",
                "SUMMARY": "summary"
            }.get(data.get("type", "GAUGE"), "gauge")
            lines.append(f"# TYPE {name} {prom_type}")

            # Labels
            labels = data.get("labels", {})
            label_str = ""
            if labels:
                label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"

            # Value
            if prom_type == "histogram":
                # Histogram buckets
                for bucket, count in sorted(data.get("buckets", {}).items()):
                    lines.append(f'{name}_bucket{{le="{bucket}"}}{label_str} {count}')
                lines.append(f'{name}_bucket{{le="+Inf"}}{label_str} {data.get("count", 0)}')
                lines.append(f"{name}_sum{label_str} {data.get('sum', 0)}")
                lines.append(f"{name}_count{label_str} {data.get('count', 0)}")
            else:
                lines.append(f"{name}{label_str} {data.get('value', 0)}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.collector.collect(), indent=2)

    def start_periodic_export(
        self,
        endpoint: str,
        interval: float = 10.0
    ) -> None:
        """Start periodic metric export."""
        self._export_interval = interval
        self._running = True
        self._export_thread = threading.Thread(
            target=self._export_loop,
            args=(endpoint,),
            daemon=True
        )
        self._export_thread.start()

    def stop_periodic_export(self) -> None:
        """Stop periodic export."""
        self._running = False
        if self._export_thread:
            self._export_thread.join(timeout=5.0)

    def _export_loop(self, endpoint: str) -> None:
        """Background export loop."""
        while self._running:
            try:
                self._export_to_endpoint(endpoint)
            except Exception:
                pass
            time.sleep(self._export_interval)

    def _export_to_endpoint(self, endpoint: str) -> None:
        """Export to endpoint (stub for actual implementation)."""
        # Would send to Prometheus pushgateway, StatsD, etc.
        pass


# Pre-configured surgical robot metrics
def create_surgical_metrics() -> MetricsCollector:
    """Create standard surgical robot metrics."""
    collector = MetricsCollector("surgical_robot")

    # Robot metrics
    collector.register_gauge("joint_position", "Joint position", "radians")
    collector.register_gauge("joint_velocity", "Joint velocity", "rad/s")
    collector.register_gauge("joint_torque", "Joint torque", "Nm")
    collector.register_gauge("end_effector_x", "End effector X position", "m")
    collector.register_gauge("end_effector_y", "End effector Y position", "m")
    collector.register_gauge("end_effector_z", "End effector Z position", "m")
    collector.register_gauge("contact_force", "Contact force", "N")

    # Control metrics
    collector.register_histogram(
        "control_loop_time",
        "Control loop execution time",
        [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    )
    collector.register_gauge("tracking_error", "Position tracking error", "m")

    # Safety metrics
    collector.register_counter("safety_violations", "Safety violations count")
    collector.register_counter("emergency_stops", "Emergency stop count")
    collector.register_gauge("safety_margin", "Safety margin percentage", "%")

    # Communication metrics
    collector.register_counter("packets_sent", "Packets sent")
    collector.register_counter("packets_received", "Packets received")
    collector.register_counter("packet_errors", "Packet errors")
    collector.register_histogram(
        "communication_latency",
        "Communication latency",
        [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    )

    # Procedure metrics
    collector.register_counter("procedure_count", "Procedures performed")
    collector.register_histogram(
        "procedure_duration",
        "Procedure duration",
        [300, 600, 1200, 1800, 3600, 7200]  # 5min to 2hr
    )

    return collector
