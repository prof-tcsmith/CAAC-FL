"""
Aggregators module
Provides various federated learning aggregation methods
"""

from .base import Aggregator
from .fedavg import FedAvgAggregator
from .krum import KrumAggregator
from .fltrust import FLTrustAggregator
from .caac_fl import CAACFLAggregator, CAACFLConfig
from .trimmed_mean import TrimmedMeanAggregator
from .median import MedianAggregator

from typing import Dict, Any


def create_aggregator(name: str, config: Dict[str, Any]) -> Aggregator:
    """
    Factory function to create aggregators.
    
    Args:
        name: Name of aggregator (fedavg, krum, fltrust, caac_fl, etc.)
        config: Configuration dictionary
        
    Returns:
        Aggregator instance
    """
    name = name.lower()
    
    if name == "fedavg":
        return FedAvgAggregator()
    elif name == "krum":
        return KrumAggregator(
            num_byzantine=config.get("num_byzantine", 0),
            num_selected=config.get("num_selected", 1)
        )
    elif name == "fltrust":
        return FLTrustAggregator(
            root_dataset_size=config.get("root_dataset_size", 100)
        )
    elif name == "caac_fl":
        caac_config = CAACFLConfig(
            beta=config.get("beta", 0.9),
            gamma=config.get("gamma", 0.1),
            lambda_mag=config.get("lambda_mag", 0.4),
            lambda_dir=config.get("lambda_dir", 0.4),
            lambda_temp=config.get("lambda_temp", 0.2),
            tau_anomaly=config.get("tau_anomaly", 2.0),
            bootstrap_rounds=config.get("bootstrap_rounds", 10),
        )
        return CAACFLAggregator(config=caac_config)
    elif name == "trimmed_mean":
        return TrimmedMeanAggregator(
            trim_ratio=config.get("trim_ratio", 0.1)
        )
    elif name == "median":
        return MedianAggregator()
    else:
        raise ValueError(f"Unknown aggregator: {name}")


__all__ = [
    "Aggregator",
    "FedAvgAggregator",
    "KrumAggregator",
    "FLTrustAggregator",
    "CAACFLAggregator",
    "CAACFLConfig",
    "TrimmedMeanAggregator",
    "MedianAggregator",
    "create_aggregator",
]