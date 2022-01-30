from sklearn.base import (
    BaseEstimator,
    BiclusterMixin,
    ClassifierMixin,
    ClusterMixin,
    DensityMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.feature_selection import SelectorMixin
from sklearn_protocols.base import (
    BaseEstimatorProtocol,
    BiclusterMixinProtocol,
    ClassifierMixinProtocol,
    ClusterMixinProtocol,
    DensityMixinProtocol,
    RegressorMixinProtocol,
    TransformerMixinProtocol,
    SelectorMixinProtocol,
)


def test_base_estimator_protocol():
    # Check whether BaseEstimator satisfies BaseEstimatorProtocol
    assert issubclass(BaseEstimator, BaseEstimatorProtocol)
    # Check whether BaseEstimatorProtocol has the public attributes of BaseEstimator  # noqa
    for attr in dir(BaseEstimator):
        if not attr.startswith("_"):
            hasattr(BaseEstimatorProtocol, attr)


def test_bicluster_mixin_protocol():
    # Check whether BiclusterMixin satisfies BiclusterMixinProtocol
    assert issubclass(BiclusterMixin, BiclusterMixinProtocol)
    # Check whether BiclusterMixinProtocol has the public attributes of BiclusterMixin  # noqa
    for attr in dir(BiclusterMixin):
        if not attr.startswith("_"):
            hasattr(BiclusterMixinProtocol, attr)


def test_classifier_mixin_protocol():
    # Check whether ClassifierMixin satisfies ClassifierMixinProtocol
    assert issubclass(ClassifierMixin, ClassifierMixinProtocol)
    # Check whether ClassifierMixinProtocol has the public attributes of ClassifierMixin  # noqa
    for attr in dir(ClassifierMixin):
        if not attr.startswith("_"):
            hasattr(ClassifierMixinProtocol, attr)


def test_cluster_mixin_protocol():
    # Check whether ClusterMixin satisfies ClusterMixinProtocol
    assert issubclass(ClusterMixin, ClusterMixinProtocol)
    # Check whether ClusterMixinProtocol has the public attributes of ClusterMixin  # noqa
    for attr in dir(ClusterMixin):
        if not attr.startswith("_"):
            hasattr(ClusterMixinProtocol, attr)


def test_density_mixin_protocol():
    # Check whether DensityMixin satisfies DensityMixinProtocol
    assert issubclass(DensityMixin, DensityMixinProtocol)
    # Check whether DensityMixinProtocol has the public attributes of DensityMixin  # noqa
    for attr in dir(DensityMixin):
        if not attr.startswith("_"):
            hasattr(DensityMixinProtocol, attr)


def test_regressor_mixin_protocol():
    # Check whether RegressorMixin satisfies RegressorMixinProtocol
    assert issubclass(RegressorMixin, RegressorMixinProtocol)
    # Check whether RegressorMixinProtocol has the public attributes of RegressorMixin  # noqa
    for attr in dir(RegressorMixin):
        if not attr.startswith("_"):
            hasattr(RegressorMixinProtocol, attr)


def test_transformer_mixin_protocol():
    # Check whether TransformerMixin satisfies TransformerMixinProtocol
    assert issubclass(TransformerMixin, TransformerMixinProtocol)
    # Check whether TransformerMixinProtocol has the public attributes of TransformerMixin  # noqa
    for attr in dir(TransformerMixin):
        if not attr.startswith("_"):
            hasattr(TransformerMixinProtocol, attr)


def test_selector_mixin_protocol():
    # Check whether SelectorMixin satisfies SelectorMixinProtocol
    assert issubclass(SelectorMixin, SelectorMixinProtocol)
    # Check whether SelectorMixinProtocol has the public attributes of SelectorMixin  # noqa
    for attr in dir(SelectorMixin):
        if not attr.startswith("_"):
            hasattr(SelectorMixinProtocol, attr)
