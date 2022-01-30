from typing import Protocol, runtime_checkable
from .base import (
    BaseEstimatorProtocol,
    # BiclusterMixinProtocol,
    ClassifierMixinProtocol,
    ClusterMixinProtocol,
    # DensityMixinProtocol,
    RegressorMixinProtocol,
    TransformerMixinProtocol,
    # SelectorMixinProtocol,
)

from .support import (
    SupportFit,
    SupportPredict,
    SupportPredictProba,
    SupportTransform,
    # SupportFitPredict,
    SupportFitTransform,
)

__all__ = [
    "RegressorProtocol",
    "ClassifierProtocol",
    "ClassifierWithPredictProbaProtocol",
    "TransformerProtocol",
    "ClusterProtocol",
]


@runtime_checkable
class RegressorProtocol(
    BaseEstimatorProtocol,
    RegressorMixinProtocol,
    SupportFit,
    SupportPredict,
    Protocol,
):
    ...


@runtime_checkable
class ClassifierProtocol(
    BaseEstimatorProtocol,
    ClassifierMixinProtocol,
    SupportFit,
    SupportPredict,
    Protocol,
):
    ...


@runtime_checkable
class ClassifierWithPredictProbaProtocol(
    BaseEstimatorProtocol,
    ClassifierMixinProtocol,
    SupportFit,
    SupportPredict,
    SupportPredictProba,
    Protocol,
):
    ...


@runtime_checkable
class TransformerProtocol(
    BaseEstimatorProtocol,
    TransformerMixinProtocol,
    SupportFit,
    SupportTransform,
    SupportFitTransform,
    Protocol,
):
    ...


@runtime_checkable
class ClusterProtocol(
    BaseEstimatorProtocol,
    ClusterMixinProtocol,
    SupportFit,
    Protocol,
):
    ...
