"""
Visualization and analysis tools for transformer-based flare detection.
"""

from .attention_visualizer import AttentionVisualizer
from .metrics_evaluation import MetricsEvaluator
from .error_analysis import ErrorAnalyzer

__all__ = [
    'AttentionVisualizer',
    'MetricsEvaluator',
    'ErrorAnalyzer',
]