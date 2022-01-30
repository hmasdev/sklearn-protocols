from typing import Protocol, runtime_checkable

__all__ = [
    "SupportFit",
    "SupportPredict",
    "SupportPredictProba",
    "SupportPredictLogProba",
    "SupportTransform",
    "SupportFitPredict",
    "SupportFitTransform",
]


@runtime_checkable
class SupportFit(Protocol):

    def fit(self, X, y=None, **kwargs):
        ...


@runtime_checkable
class SupportPredict(Protocol):

    def predict(self, X, **kwargs):
        ...


@runtime_checkable
class SupportPredictProba(Protocol):

    def predict_proba(self, X, **kwargs):
        ...


@runtime_checkable
class SupportPredictLogProba(Protocol):

    def predict_log_proba(self, X, **kwargs):
        ...


@runtime_checkable
class SupportTransform(Protocol):

    def transform(self, X, y=None, **kwargs):
        ...


@runtime_checkable
class SupportFitPredict(Protocol):

    def fit_predict(self, X, y=None, **kwargs):
        ...


@runtime_checkable
class SupportFitTransform(Protocol):

    def fit_transform(self, X, y=None, **kwargs):
        ...
