import pytest
from sklearn.utils import all_estimators
from sklearn_protocols import (
    RegressorProtocol,
    ClassifierProtocol,
    ClassifierWithPredictProbaProtocol,
    TransformerProtocol,
    ClusterProtocol,
)


# NOTE: 'type_filter' in all_estimators is based on Mixin,
#       like RegressorMixin, ClassifierMixin and so on.
regressors = all_estimators("regressor")
classifiers = all_estimators("classifier")
transformers = all_estimators("transformer")
clusters = all_estimators("cluster")
regressors_classifiers = [
    (name, cls_)
    for name, cls_ in all_estimators()
    if name in (
        # These are used as a regressor or a classifier.
        'GridSearchCV',
        'HalvingGridSearchCV',
        'HalvingRandomSearchCV',
        'Pipeline',
        'RandomizedSearchCV',
    )
]
regressors += regressors_classifiers
classifiers += regressors_classifiers


def public_dir(obj):
    return [x for x in dir(obj) if not x.startswith("_")]


def comp_public_dirs(cls, proto):
    return f"cls has {public_dir(cls)} but proto has {public_dir(proto)}"


@pytest.mark.parametrize("name,cls", regressors)
def test_regressor_protocol(name: str, cls: type):
    assert issubclass(cls, RegressorProtocol), comp_public_dirs(cls, RegressorProtocol)  # noqa


@pytest.mark.parametrize("name,cls", classifiers)
def test_classifier_protocol(name: str, cls: type):

    classifiers_without_predict_proba = (
        "LinearSVC",
        "NearestCentroid",
        "OneVsOneClassifier",
        "OutputCodeClassifier",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "RidgeClassifier",
        "RidgeClassifierCV",
    )

    if name in classifiers_without_predict_proba:
        assert issubclass(cls, ClassifierProtocol), comp_public_dirs(cls, ClassifierWithPredictProbaProtocol)  # noqa
    else:
        assert issubclass(cls, ClassifierProtocol), comp_public_dirs(cls, ClassifierWithPredictProbaProtocol)  # noqa
        assert issubclass(cls, ClassifierWithPredictProbaProtocol), comp_public_dirs(cls, ClassifierProtocol)  # noqa


@pytest.mark.parametrize("name,cls", transformers)
def test_transformer_protocol(name: str, cls: type):
    assert issubclass(cls, TransformerProtocol), comp_public_dirs(cls, TransformerProtocol)  # noqa


@pytest.mark.parametrize("name,cls", clusters)
def test_cluster_protocol(name: str, cls: type):
    assert issubclass(cls, ClusterProtocol), comp_public_dirs(cls, ClusterProtocol)  # noqa
