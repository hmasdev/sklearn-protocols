from typing import Protocol, runtime_checkable

__all__ = [
    "BaseEstimatorProtocol",
    "BiclusterMixinProtocol",
    "ClassifierMixinProtocol",
    "ClusterMixinProtocol",
    "DensityMixinProtocol",
    "RegressorMixinProtocol",
    "TransformerMixinProtocol",
    "SelectorMixinProtocol",
]


@runtime_checkable
class BaseEstimatorProtocol(Protocol):

    def get_params(self, deep=False):
        ...

    def set_params(self, **params):
        ...


@runtime_checkable
class BiclusterMixinProtocol(Protocol):

    def get_indices(self, i):
        ...

    def get_shape(self, i):
        ...

    def get_submatrix(self, i, data):
        ...


@runtime_checkable
class ClassifierMixinProtocol(Protocol):

    def score(self, X, y, sample_weight=None):
        ...


@runtime_checkable
class ClusterMixinProtocol(Protocol):

    def fit_predict(self, X, y=None):
        ...


@runtime_checkable
class DensityMixinProtocol(Protocol):

    def score(self, X, y=None):
        ...


@runtime_checkable
class RegressorMixinProtocol(Protocol):

    def score(X, y, sample_weight=None):
        ...


@runtime_checkable
class TransformerMixinProtocol(Protocol):

    def fit_transform(self, X, y=None, **fit_params):
        ...


@runtime_checkable
class SelectorMixinProtocol(Protocol):

    def fit_transform(self, X, y=None):
        ...

    def get_feature_names_out(self, input_features=None):
        ...

    def get_support(self, indices=False):
        ...

    def inverse_transform(self, X):
        ...

    def transform(self, X):
        ...
