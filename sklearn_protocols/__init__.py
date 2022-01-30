from .model import (
    RegressorProtocol,
    ClassifierProtocol,
    ClassifierWithPredictProbaProtocol,
    TransformerProtocol,
    ClusterProtocol,
)

__version__ = "0.0"

__all__ = [
    RegressorProtocol.__name__,
    ClassifierProtocol.__name__,
    ClassifierWithPredictProbaProtocol.__name__,
    TransformerProtocol.__name__,
    ClusterProtocol.__name__,
]
