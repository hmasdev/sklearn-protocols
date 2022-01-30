from sklearn_protocols.model import (
    RegressorProtocol,
    ClassifierProtocol,
    ClassifierWithPredictProbaProtocol,
    TransformerProtocol,
    ClusterProtocol,
)


def test_regressor_protocol():
    hasattr(RegressorProtocol, "get_params")
    hasattr(RegressorProtocol, "set_params")
    hasattr(RegressorProtocol, "fit")
    hasattr(RegressorProtocol, "predict")
    hasattr(RegressorProtocol, "score")


def test_classifier_protocol():
    hasattr(ClassifierProtocol, "get_params")
    hasattr(ClassifierProtocol, "set_params")
    hasattr(ClassifierProtocol, "fit")
    hasattr(ClassifierProtocol, "predict")
    hasattr(ClassifierProtocol, "score")


def test_classifier_with_predict_proba_protocol():
    hasattr(ClassifierWithPredictProbaProtocol, "get_params")
    hasattr(ClassifierWithPredictProbaProtocol, "set_params")
    hasattr(ClassifierWithPredictProbaProtocol, "fit")
    hasattr(ClassifierWithPredictProbaProtocol, "predict")
    hasattr(ClassifierWithPredictProbaProtocol, "predict_proba")
    hasattr(ClassifierWithPredictProbaProtocol, "score")


def test_transformer_protocol():
    hasattr(TransformerProtocol, "get_params")
    hasattr(TransformerProtocol, "set_params")
    hasattr(TransformerProtocol, "fit")
    hasattr(TransformerProtocol, "transform")
    hasattr(TransformerProtocol, "fit_transform")


def test_cluster_protocol():
    hasattr(ClusterProtocol, "get_params")
    hasattr(ClusterProtocol, "set_params")
    hasattr(ClusterProtocol, "fit")
    hasattr(ClusterProtocol, "fit_predict")
